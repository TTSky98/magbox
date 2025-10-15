from magbox.heff import heff
import torch 

class llg:
    def __init__(self,sp,gamma=1, alpha=0.01, Temp=0, dt=0.1, T=50):
        data_type=sp.data_type
        device=sp.device
        self.num=sp.num
        self.gamma=torch.tensor(gamma,dtype=data_type,device=device)
        self.alpha=torch.tensor(alpha,dtype=data_type,device=device)
        self.Temp=torch.tensor(Temp,dtype=data_type,device=device)
        self.dt=torch.tensor(dt,dtype=data_type,device=device)
        self.T=torch.tensor(T,dtype=data_type,device=device)

        self.block_crow=torch.arange(0,self.num+1,dtype=torch.int64,device=device)
        self.block_col=torch.arange(0,self.num,dtype=torch.int64,device=device)

        self.h_fun=heff(sp)

    def llg_kernal(self,sp):
        cscv=1/sp.s_theta
        cscv2=cscv**2
        value=self.gamma/(1+self.alpha**2)*torch.cat([-self.alpha*torch.ones(self.num,1,dtype=sp.data_type,device=sp.device),-cscv,cscv,-self.alpha*cscv2],1)
        return torch.sparse_bsr_tensor(self.block_crow,self.block_col,value.reshape(-1,2,2))
    
    def llg_convert(self,sp):
        value=-torch.cat([-sp.s_theta*sp.s_phi , sp.s_theta*sp.c_phi,torch.zeros(sp.num,1,dtype=sp.data_type,device=sp.device),
                           sp.c_theta*sp.c_phi, sp.c_theta*sp.s_phi, -sp.s_theta],1)
        return torch.sparse_bsr_tensor(self.block_crow,self.block_col,value.reshape(-1,2,3),dtype=sp.data_type,device=sp.device)

    def llg_drift(self, sp):
        kernal=self.llg_kernal(sp)
        h3_to_h2=self.llg_convert(sp)
        h=self.h_fun.all3(sp)
        return kernal @ (h3_to_h2 @ h)
