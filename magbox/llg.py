from magbox.heff import heff
import torch 
from magbox import boxlib

class llg:
    def __init__(self,sp,vars:dict={},gamma=1, alpha=0.01, Temp=0, dt=0.1, T=50):
        data_type=sp.data_type
        device=sp.device
        self.num=sp.num
        self.gamma=torch.tensor(gamma,dtype=data_type,device=device)
        self.alpha=torch.tensor(alpha,dtype=data_type,device=device)
        self.Temp=torch.tensor(Temp,dtype=data_type,device=device)
        self.dt=torch.tensor(dt,dtype=data_type,device=device)
        self.T=torch.tensor(T,dtype=data_type,device=device)
        self.tspan=torch.linspace(0,self.T,int(self.T/self.dt)+1,dtype=data_type,device=device)
        self.h_fun=heff(sp,vars)
        self.data_type=sp.data_type
        self.device=sp.device

        # prepare sparse matrix index for BSR format
        self.block_crow=torch.arange(0,self.num+1,dtype=torch.int64,device=device)
        self.block_col=torch.arange(0,self.num,dtype=torch.int64,device=device)

        # prepare sparse matrix index for CSR format
        odd_pos = torch.arange((self.num*2 + 2) // 2,dtype=torch.int64,device=device) * 5
        even_pos = odd_pos[:-1] + 3
        self.csr_crow=torch.zeros(self.num*2+1, dtype=torch.int64,device=device)
        self.csr_crow[0::2] = odd_pos
        self.csr_crow[1::2] = even_pos

        total_length = 5 * self.num
        group_indices = torch.arange(total_length,dtype=torch.int64,device=device) // 5
        pos_in_group = torch.arange(total_length,dtype=torch.int64,device=device) % 5
        col_pattern=torch.tensor([0,1,2,0,1],dtype=torch.int64,device=device)
        self.csr_col=group_indices * 3 + col_pattern[pos_in_group]



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
        return torch.sparse_csr_tensor(self.csr_crow,self.csr_col,value.reshape(-1),(self.num*2,self.num*3),dtype=self.data_type,device=self.device)

    def llg_drift(self,t, ang):
        theta=ang[0::2]
        phi=ang[1::2]

        s_theta=torch.sin(theta)
        c_theta=torch.cos(theta)
        s_phi=torch.sin(phi)
        c_phi=torch.cos(phi)

        kernal=self.llg_kernal(s_theta)
        h3_to_h2=self.llg_convert(s_theta,c_theta,s_phi,c_phi)
        h3=self.h_fun.all3(s_theta,c_theta,s_phi,c_phi)
        return kernal @ (h3_to_h2 @ h3)
    def run_zero_Temp(self,sp):
        llg_fun=lambda t, y: self.llg_drift(t,y)
        
        ini=sp.ang
        odeset={"rel_tol":max(self.alpha*1e-2,1e-6),"abs_tol":max(self.alpha*1e-4,1e-6)}
        t,ang,stats,erro_info=boxlib.ode_rk45(llg_fun,self.tspan,ini,options=odeset)
        return t,ang,stats,erro_info
