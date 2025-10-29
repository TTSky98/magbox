from .heff import heff
import torch 
from . import boxlib
from .initial import Lattice, Vars
from .spin import spin
import warnings
from typing import Union, Tuple
import torch

def safe_sin(theta: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    mask = torch.abs(sin_theta) < eps
    safe_cos = torch.where(
        mask,
        torch.ones_like(sin_theta) * eps,
        sin_theta
    )
    return safe_cos

class llg:
    def __init__(self,sp:spin,vars:Vars=Vars(),gamma=1, alpha=0.01, Temp=0., dt=0.1, T=50, rtol:Union[float,None]=None, atol:Union[float,None]=None):
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        self.rtol=rtol
        self.atol=atol
        data_type=sp.dtype
        device=sp.device
        self.num=sp.num
        self.gamma=torch.tensor(gamma,dtype=data_type,device=device)
        self.alpha=torch.tensor(alpha,dtype=data_type,device=device)
        self.Temp=torch.tensor(Temp,dtype=data_type,device=device)
        self.dt=torch.tensor(dt,dtype=data_type,device=device)
        self.T=torch.tensor(T,dtype=data_type,device=device)
        self.tspan=torch.linspace(0,self.T,int(self.T/self.dt)+1,dtype=data_type,device=device)
        self.h_fun=heff(sp,vars)
        self.data_type=sp.dtype
        self.device=sp.device
        self.eps=torch.finfo(self.data_type).eps

        # strength for thermal field
        self.thermal_strength=torch.sqrt(2*self.alpha*self.Temp/self.gamma)

        # prepare sparse matrix index for BSR format
        self.block_crow=torch.arange(0,self.num+1,dtype=torch.int64,device=device)
        self.block_col=torch.arange(0,self.num,dtype=torch.int64,device=device)

        # prepare sparse matrix index for CSR format 
        '''
        format:[[*,*,*]
                [*,*,0]]
        '''
        odd_pos = torch.arange((self.num*2 + 2) // 2,dtype=torch.int64,device=device) * 5
        even_pos = odd_pos[:-1] + 3
        self.convert_csr_crow=torch.zeros(self.num*2+1, dtype=torch.int64,device=device)
        self.convert_csr_crow[0::2] = odd_pos
        self.convert_csr_crow[1::2] = even_pos

        total_length = 5 * self.num
        group_indices = torch.arange(total_length,dtype=torch.int64,device=device) // 5
        pos_in_group = torch.arange(total_length,dtype=torch.int64,device=device) % 5
        col_pattern=torch.tensor([0,1,2,0,1],dtype=torch.int64,device=device)
        self.convert_csr_col=group_indices * 3 + col_pattern[pos_in_group]

        # prepare sparse matrix index for CSR format for drift cart to sph
        '''
        format:[[*,*,*]
                [*,*,*]]
        '''
        self.drift_cart_to_sph_csr_crow=torch.arange(0,self.num*6+1,step=3,dtype=torch.int64,device=device)
        
        total_length = 6 * self.num
        group_indices = torch.arange(total_length,dtype=torch.int64,device=device) // 6
        pos_in_group = torch.arange(total_length,dtype=torch.int64,device=device) % 6
        col_pattern=torch.tensor([0,1,2,0,1,2],dtype=torch.int64,device=device)
        self.drift_cart_to_sph_csr_col=group_indices * 3 + col_pattern[pos_in_group]

    def llg_kernal(self,s_theta):
        cscv=1/s_theta
        cscv2=cscv**2
        value=self.gamma/(1+self.alpha**2)*torch.cat([-self.alpha*torch.ones(self.num,1,dtype=self.data_type,device=self.device),-cscv,cscv,-self.alpha*cscv2],1)
        return torch.sparse_bsr_tensor(self.block_crow,self.block_col,value.reshape(-1,2,2),(self.num*2,self.num*2))
    
    def llg_convert_bsr(self,sp):
        value=-torch.cat([sp.c_theta*sp.c_phi, sp.c_theta*sp.s_phi, -sp.s_theta,
                          -sp.s_theta*sp.s_phi , sp.s_theta*sp.c_phi,torch.zeros(sp.num,1,dtype=sp.data_type,device=sp.device)],1)
        return torch.sparse_bsr_tensor(self.block_crow,self.block_col,value.reshape(-1,2,3),(self.num*2,self.num*3),dtype=sp.data_type,device=sp.device)
    def llg_convert(self,s_theta,c_theta,s_phi,c_phi):
        value=-torch.cat([c_theta*c_phi, c_theta*s_phi, -s_theta,
                            -s_theta*s_phi , s_theta*c_phi],1)
        return torch.sparse_csr_tensor(self.convert_csr_crow,self.convert_csr_col,value.reshape(-1),(self.num*2,self.num*3),dtype=self.data_type,device=self.device)
    
    def llg_drift_cart_to_sph(self,s_theta,c_theta,s_phi,c_phi):
        cot_theta=c_theta/s_theta
        csc_theta=1/s_theta
        value=self.gamma/(1+self.alpha**2)*torch.cat(
            [self.alpha*c_theta*c_phi-s_phi, c_phi+self.alpha*c_theta*s_phi, -self.alpha*s_theta,
             -self.alpha*csc_theta*s_phi-cot_theta*c_phi, -cot_theta*s_phi + self.alpha*csc_theta*c_phi, torch.ones(self.num,1,dtype=self.data_type,device=self.device)],1)
        return torch.sparse_csr_tensor(self.drift_cart_to_sph_csr_crow,self.drift_cart_to_sph_csr_col,value.reshape(-1), (self.num*2, self.num*3),dtype=self.data_type,device=self.device)

    def llg_drift(self,t, ang):
        theta=ang[0::2]
        phi=ang[1::2]

        s_theta=safe_sin(theta, self.eps)
        c_theta=torch.cos(theta)
        s_phi=torch.sin(phi)
        c_phi=torch.cos(phi)

        # kernal=self.llg_kernal(s_theta)
        # h3_to_h2=self.llg_convert(s_theta,c_theta,s_phi,c_phi)
        kernal=self.llg_drift_cart_to_sph(s_theta,c_theta,s_phi,c_phi)
        h3=self.h_fun.all3(s_theta,c_theta,s_phi,c_phi)
        drift_core=kernal @ h3
        return drift_core
    def llg_thermal(self, t, ang):
        theta=ang[0::2]
        phi=ang[1::2]

        s_theta=safe_sin(theta, self.eps)
        c_theta=torch.cos(theta)
        s_phi=torch.sin(phi)
        c_phi=torch.cos(phi)

        # kernal=self.llg_kernal(s_theta)
        # h3_to_h2=self.llg_convert(s_theta,c_theta,s_phi,c_phi)
        kernal=self.llg_drift_cart_to_sph(s_theta,c_theta,s_phi,c_phi)
        h3=self.h_fun.all3(s_theta,c_theta,s_phi,c_phi)
        correction=self.Stratonovich_correction(s_theta,c_theta)
        drift_core=kernal @ h3
        return drift_core-correction, kernal*self.thermal_strength

    def Stratonovich_correction(self,s_theta,c_theta):
        cot_theta=c_theta/s_theta
        value=self.gamma * self.alpha * self.Temp/(1+self.alpha**2)*torch.cat([cot_theta, torch.zeros(self.num,1,dtype=self.data_type,device=self.device)],1)
        return value.reshape(-1,1)
    def run(self,sp)-> Tuple[torch.Tensor,torch.Tensor,dict, dict]:
        # error control
        if self.rtol is None:
            if self.Temp ==0 :
                rtol=max(self.alpha.item()*1e-2,1e-3)
            else:
                rtol=max(self.alpha.item()/5,1e-2)
        else:
            rtol=self.rtol

        if self.atol is None:
            if self.Temp ==0 :
                atol=max(self.alpha.item()*1e-4,1e-6)
            else:
                atol=max(self.alpha.item()*1e-2,1e-3)
        else:
            atol=self.atol
        odeset={"rel_tol":rtol,"abs_tol":atol}
        ini=sp.ang
        if self.Temp==0:
            llg_fun=self.llg_drift
            t,ang,stats,erro_info=boxlib.ode_rk45(llg_fun, self.tspan, ini, options=odeset)
        else:
            llg_fun=self.llg_thermal
            t,ang,stats,erro_info=boxlib.ode_sde_em(llg_fun, self.tspan,ini, options=odeset)
        return t,ang,stats,erro_info
