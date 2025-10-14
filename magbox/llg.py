from magbox import heff,boxlib
import torch 

class llg:
    def __init__(self,sp,gamma=1, alpha=0.01, Temp=0, dt=0.1, T=50,type="f32"):
        data_type=boxlib.get_data_type(type)
        self.num=sp.num
        self.gamma=torch.tensor(gamma,dtype=data_type)
        self.alpha=torch.tensor(alpha,dtype=data_type)
        self.Temp=torch.tensor(Temp,dtype=data_type)
        self.dt=torch.tensor(dt,dtype=data_type)
        self.T=torch.tensor(T,dtype=data_type)

        self.block_crow=torch.range(0,self.num,dtype=torch.int64)
        self.block_col=torch.range(0,self.num-1,dtype=torch.int64)

    def llg_kernal(self,sp):
        cscv=1/torch.sin(sp.theta)
        cscv2=cscv**2
        value=self.gamma/(1+self.alpha**2)*torch.cat([-self.alpha*torch.ones(self.num,1),-cscv,cscv,-self.alpha*cscv2],1)
        return torch.sparse_bsc_tensor(self.block_crow,self.block_col,value.reshape(-1,2,2))

    def llg_drift(self, sp):
        kernal=self.llg_kernal(sp)
        h=heff.all(sp)
        return kernal@h
