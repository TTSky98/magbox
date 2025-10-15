import torch
from magbox import boxlib

class heff:
    def __init__(self, spin, B=0,B_dir=[0,0,1],J=1,K_dir=[0,0,1],K1=1,D=0):
        # self.spin = spin
        self.num=spin.num
        self.B=torch.tensor(B,dtype=spin.data_type,device=spin.device)# external field
        self.B_dir=torch.tensor(B_dir,dtype=spin.data_type,device=spin.device)
        self.B_dir=self.B_dir.view(3)
        self.B_dir=self.B_dir/torch.norm(self.B_dir)
        self.B=self.B*self.B_dir

        self.K_dir=torch.tensor(K_dir,dtype=spin.data_type,device=spin.device)
        self.K_dir=self.K_dir.view(3)
        self.K_dir=self.K_dir/torch.norm(self.K_dir)
        self.K1=torch.tensor(K1,dtype=spin.data_type,device=spin.device)
        self.K1=self.K1*self.K_dir

        self.H = torch.zeros((self.num,3))
        self.data_type=spin.data_type
        self.device=spin.device

        l_type=spin.lattice_type
        N_dim=l_type.get("size",[])
        N_dim=len(N_dim)

        if isinstance(J,(int,float)) or (isinstance(J,list) and len(J)==1):
            J_val=J
            self.Jmtx=J_val*boxlib.get_Jmtx(spin.lattice_type, device=spin.device,data_type=spin.data_type)
        elif isinstance(J,list) and len(J)==N_dim:
            self.Jmtx=J.to(device=spin.device,dtype=spin.data_type)
    def zeeman3(self):
        return self.B*torch.ones((self.num,3),dtype=self.data_type,device=self.device)
    # def zeeman2(self,ctheta,stheta,cphi,sphi):
    #     h_theta=self.B_dir@torch.cat([])
    # return self.B*torch.cat([*],1)

    def dipole3(self):
        # calculate dipole field
        return torch.zeros((self.num,3))

    def exchange3(self,sp):
        # calculate exchange field
        return self.Jmtx@sp.cart_S
    def uni_anisotropy3(self,sp):
        # calculate anisotropy field
        return self.K1*sp.cart_S
    def DM3(self):
        # calculate DM field
        return torch.zeros((self.num,3))
    def all3(self, sp ):
        h0=self.zeeman3() + self.exchange3(sp) + self.uni_anisotropy3(sp)
        return h0.reshape(-1,1)