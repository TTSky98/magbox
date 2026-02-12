from .heff3 import heff3
import torch 
from . import boxlib
from .initial import Lattice, Vars
from .spin3 import spin3
import warnings
from typing import Union, Tuple

class llg3:
    def __init__(self,x,y,z, lattice_type:Lattice,vars:Vars=Vars(),
                 dtype="f32", device="gpu", thread:int=4, require_ini_grad:bool=False,
                 gamma=1., alpha=0.01, Temp=0., dt=0.1, T=50., rtol:Union[float,None]=None, atol:Union[float,None]=None):
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        sp = spin3(x,y,z,lattice_type,dtype=dtype, device=device, thread=thread, require_ini_grad=require_ini_grad)
        self.spin = sp

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

    def llg_drift(self,t, S):
        x,y,z=spin3.get_xyz(S.view(-1,1))
        h3=self.h_fun.all3(t,x,y,z).view(-1,3)
        M1=torch.linalg.cross(S,h3,dim=1)
        M2=torch.linalg.cross(S,M1,dim=1)
        drift_core=M1 + self.alpha * M2
        return self.prefactor*drift_core
    
    def llg_thermal_no_correction(self, t, S):
        x,y,z=spin3.get_xyz(S.view(-1,1))
        h3=self.h_fun.all3(t,x,y,z).view(-1,3) 
        dim=h3.shape
        M1=torch.linalg.cross(S,h3,dim=1)
        M2=torch.linalg.cross(S,M1,dim=1)
        f=self.prefactor * (M1 + self.alpha * M2)
        def gfun(r):
            gM1=torch.linalg.cross(S,r,dim=1)
            gM2=torch.linalg.cross(S,gM1,dim=1)
            return self.thermal_strength*self.prefactor * (gM1 + self.alpha * gM2)
        return f, gfun, dim
    def llg_thermal(self, t, S):
        f, g, dim =self.llg_thermal_no_correction(t,S)
        correction = self.Stratonovich_correction(S) 
        return f-correction, g, dim
    def Stratonovich_correction(self,S):
        return 2*self.prefactor*self.alpha*self.Temp*S
    def run(self,ini=None, waitbar:bool=True)-> Tuple[torch.Tensor,torch.Tensor,dict, dict]:
        # error control
        if ini is None:
            ini=self.spin.cart_S
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
        odeset={"rel_tol":rtol,"abs_tol":atol, "waitbar":waitbar}
        
        if self.Temp==0:
            llg_fun=self.llg_drift
            t,Sout,stats,erro_info=boxlib.ode3_rk45(llg_fun, self.tspan, ini, options=odeset)
        else:
            llg_fun=self.llg_thermal_no_correction
            t,Sout,stats,erro_info=boxlib.ode3_sde_em(llg_fun, self.tspan,ini, options=odeset)
        return t,Sout,stats,erro_info
