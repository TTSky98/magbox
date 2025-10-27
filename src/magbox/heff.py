import torch
from . import boxlib
from .initial import Vars, Lattice
from .spin import spin
from dataclasses import replace

class heff:
    def __init__(self, sp:spin, vars:Vars): # B=0,B_dir=[0,0,1],J=1,K_dir=[0,0,1],K1=1,D=0
        # self.spin = spin

        B=vars.B # external field
        B_dir=vars.B_dir # external field direction

        J=vars.J # exchange interaction
        K1_dir=vars.K1_dir # uniaxial anisotropy axis
        K1=vars.K1 # uniaxial anisotropy strength
        D=vars.D # Dzyaloshinskii-Moriya interaction

        self.num=sp.num
        self.B=torch.tensor(B,dtype=sp.data_type,device=sp.device)# external field
        self.B_dir=torch.tensor(B_dir,dtype=sp.data_type,device=sp.device)
        self.B_dir=self.B_dir.view(3)
        self.B_dir=self.B_dir/torch.norm(self.B_dir)
        self.B=self.B*self.B_dir

        self.K_dir=torch.tensor(K1_dir,dtype=sp.data_type,device=sp.device)
        self.K_dir=self.K_dir.view(3)
        self.K_dir=self.K_dir/torch.norm(self.K_dir)
        self.K1=torch.tensor(K1,dtype=sp.data_type,device=sp.device)
        self.K1=self.K1*self.K_dir

        self.H = torch.zeros((self.num,3))
        self.data_type=sp.data_type
        self.device=sp.device

        l_type=sp.lattice_type
        N_dim=l_type.size
        N_dim=len(N_dim)

        if isinstance(J,(int,float)) or (isinstance(J,list) and len(J)==1):
            J_val=J
            self.Jmtx=J_val*boxlib.get_Jmtx(replace(l_type,J_direction=None), device=sp.device,data_type=sp.data_type)
        elif isinstance(J,list):
            if len(J)!=N_dim:
                raise ValueError(f"J should be a scalar or a list of length {N_dim}")
            # self.Jmtx=J.to(device=spin.device,dtype=spin.data_type)
            self.Jmtx=torch.zeros((self.num,self.num),dtype=sp.data_type,device=sp.device)
            for i in range(N_dim):
                self.Jmtx+=J[i]*boxlib.get_Jmtx(replace(l_type,J_direction=i), device=sp.device,data_type=sp.data_type)
    def energy(self,ang):
        theta=ang[0::2]
        phi=ang[1::2]
        s_theta=torch.sin(theta)
        c_theta=torch.cos(theta)
        s_phi=torch.sin(phi)
        c_phi=torch.cos(phi)
        t_num=theta.shape[1]

        cartS=self.get_cart_S(s_theta,c_theta,s_phi,c_phi)

        E_zeeman= - torch.sum(self.B[0] * cartS[:,0:t_num]+self.B[1] * cartS[:,t_num:2*t_num]+self.B[2] * cartS[:,2*t_num:], dim=0)
        E_exch= -0.5 * torch.sum(cartS * (self.Jmtx @ cartS), dim=0)
        E_exch=E_exch[0::3] + E_exch[1::3] + E_exch[2::3]
        E_anis= -0.5 * torch.norm(self.K1) * torch.sum((cartS[:,0:t_num] *self.K_dir[0]+cartS[:,t_num:2*t_num] *self.K_dir[1]+cartS[:,2*t_num:] *self.K_dir[2])**2, dim=0)
        E_total=E_zeeman + E_exch + E_anis
        return E_total
    def zeeman3(self):
        return self.B*torch.ones((self.num,3),dtype=self.data_type,device=self.device)
    # def zeeman2(self,ctheta,stheta,cphi,sphi):
    #     h_theta=self.B_dir@torch.cat([])
    # return self.B*torch.cat([*],1)

    def dipole3(self):
        # calculate dipole field
        return torch.zeros((self.num,3))

    def exchange3(self,cartS):
        # calculate exchange field
        return self.Jmtx @ cartS
    def uni_anisotropy3(self,cartS):
        # calculate anisotropy field
        return self.K1 * cartS
    def DM3(self):
        # calculate DM field
        return torch.zeros((self.num,3))
    def all3(self, s_theta,c_theta,s_phi,c_phi):
        cartS=self.get_cart_S(s_theta,c_theta,s_phi,c_phi)
        h0=self.zeeman3() + self.exchange3(cartS) + self.uni_anisotropy3(cartS)
        return h0.reshape(-1,1)
    
    @staticmethod
    def get_cart_S(s_theta,c_theta,s_phi,c_phi):
        x = s_theta * c_phi
        y = s_theta * s_phi
        z = c_theta
        return torch.cat([x, y, z], dim=1)
    
        