from .heff3 import heff3
import torch 
from . import boxlib
from .initial import Lattice, Vars
from .spin3 import spin3
import warnings
from typing import Union, Tuple
import torch


class llg3:
    def __init__(self,sp:spin3,vars:Vars=Vars(),gamma=1, alpha=0.01, Temp=0., dt=0.1, T=50, rtol:Union[float,None]=None, atol:Union[float,None]=None):
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        self.rtol=rtol
        self.atol=atol
        dtype=sp.dtype
        device=sp.device
        self.num=sp.num

        self.gamma=torch.as_tensor(gamma,dtype=dtype,device=device)
        self.alpha=torch.as_tensor(alpha,dtype=dtype,device=device)
        self.Temp=torch.as_tensor(Temp,dtype=dtype,device=device)
        self.dt=torch.as_tensor(dt,dtype=dtype,device=device)
        self.T=torch.as_tensor(T,dtype=dtype,device=device)
        self.tspan=torch.linspace(0,self.T,int(self.T/self.dt)+1,dtype=dtype,device=device)
        self.h_fun=heff3(sp,vars)
        self.dtype=sp.dtype
        self.device=sp.device
        self.eps=torch.finfo(self.dtype).eps

        self.prefactor=-self.gamma/(1+self.alpha**2)

        # strength for thermal field
        self.thermal_strength=torch.sqrt(2*self.alpha*self.Temp/self.gamma)

        # prepare sparse matrix index for BSR format
        self.block_crow=torch.arange(0,self.num+1,dtype=torch.int64,device=device)
        self.block_col=torch.arange(0,self.num,dtype=torch.int64,device=device)

        # prepare sparse matrix index for CSR format 
        '''
        format:[[0,*,*]
                [*,0,*]
                [*,*,0]]
        '''
        total_length = 6 * self.num
        self.csr_crow=torch.arange(0,total_length+1,step=2, dtype=torch.int64,device=device)

        col_pattern=torch.tensor([1,2,0,2,0,1],dtype=torch.int64,device=device)
        group_indices = torch.arange(total_length,dtype=torch.int64,device=device) // 6
        pos_in_group = torch.arange(total_length,dtype=torch.int64,device=device) % 6
        self.csr_col=group_indices * 3 + col_pattern[pos_in_group]

    def M_cross_mat(self,x,y,z):
        value=torch.cat([-z,y,
                         z,-x,
                         -y,x],dim=1)
        return torch.sparse_csr_tensor(self.csr_crow,self.csr_col,value.reshape(-1),(self.num*3, self.num*3),device=self.device,dtype=self.dtype)

    def llg_drift(self,t, S):
        x,y,z=spin3.get_xyz(S)
        M=self.M_cross_mat(x,y,z)
        h3=self.h_fun.all3(t,x,y,z)
        drift_core=M @ h3 + self.alpha*M @ (M @ h3)
        return self.prefactor*drift_core
    def llg_thermal(self, t, S):
        x,y,z=spin3.get_xyz(S)
        M=self.M_cross_mat(x,y,z)
        h3=self.h_fun.all3(t,x,y,z)
        drift_core=M @ h3 + self.alpha*M @ (M @ h3)
        correction=self.Stratonovich_correction(S)
        return self.prefactor*drift_core-correction, self.thermal_strength*self.prefactor*drift_core

    def Stratonovich_correction(self,S):
        return self.prefactor*self.alpha*self.Temp*S
    def run(self,sp:spin3)-> Tuple[torch.Tensor,torch.Tensor,dict, dict]:
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
        ini=sp.S
        if self.Temp==0:
            llg_fun=self.llg_drift
            t,Sout,stats,erro_info=boxlib.ode3_rk45(llg_fun, self.tspan, ini, options=odeset)
        else:
            llg_fun=self.llg_thermal
            t,Sout,stats,erro_info=boxlib.ode3_sde_em(llg_fun, self.tspan,ini, options=odeset)
        return t,Sout,stats,erro_info
