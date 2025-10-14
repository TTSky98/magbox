import torch
from magbox import boxlib

class heff:
    def __init__(self, spin, B=0,B_dir=[0,0,1]):
        self.spin = spin
        self.num=spin.num
        self.B=B # external field
        self.B_dir=torch.tensor(B_dir,dtype=spin.data_type,device=spin.device)
        self.B_dir=self.B_dir.view(3)
        self.B_dir=self.B_dir/torch.norm(self.B_dir)
        self.H = torch.zeros((self.num,3))
        self.data_type=spin.data_type
        self.device=spin.device

        self.Jmtx=boxlib.get_Jmtx(spin.lattice_type, device=spin.device,data_type=spin.data_type)
    def zeeman3(self):
        return self.B*self.B_dir*torch.ones((self.num,3),dtype=self.data_type,device=self.device)
    # def zeeman2(self,ctheta,stheta,cphi,sphi):
    #     h_theta=self.B_dir@torch.cat([])
    # return self.B*torch.cat([*],1)

    def dipole3(self):
        # calculate dipole field
        return torch.zeros((self.num,3))

    def exchange3(self):
        # calculate exchange field
        return torch.zeros((self.num,3))
    def uni_anisotropy3(self):
        # calculate anisotropy field
        return torch.zeros((self.num,3))
    def DM3(self):
        # calculate DM field
        return torch.zeros((self.num,3))
    def all3(self):
        return self.zeeman3() + self.dipole3() + self.exchange3() + self.uni_anisotropy3() + self.DM3()