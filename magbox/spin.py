import numpy as np
import torch
from magbox import boxlib
class spin:
    def __init__(self, theta, phi, lattice_type,type="f32", device="gpu"):
        data_type=boxlib.get_data_type(type)
        self.data_type=data_type
        self.device=boxlib.get_device(device)
        self.theta = torch.tensor(theta,dtype=self.data_type,device=self.device)
        self.phi = torch.tensor(phi,dtype=self.data_type,device=self.device)
        self.theta=self.theta.view(-1,1)
        self.phi=self.phi.view(-1,1)
        l_theta=len(self.theta)
        l_phi=len(self.phi)
        if l_theta!=l_phi:
            raise ValueError("theta and phi must have the same length")
        self.num = l_theta

        self.c_theta=torch.cos(self.theta)
        self.s_theta=torch.sin(self.theta)
        self.c_phi=torch.cos(self.phi)
        self.s_phi=torch.sin(self.phi)

        self.x, self.y, self.z = self.cart()
        self.cart_S=torch.cat([self.x,self.y,self.z],1)
        self.lattice_type=lattice_type
    def cart(self): 
        # convert to cartesian coordinates
        x = self.s_theta * self.c_phi
        y = self.s_theta * self.s_phi
        z = self.c_theta
        return x, y, z
    def update(self):
        self.c_theta=torch.cos(self.theta)
        self.s_theta=torch.sin(self.theta)
        self.c_phi=torch.cos(self.phi)
        self.s_phi=torch.sin(self.phi)
        
        self.x, self.y, self.z = self.cart()
        self.cart_S=torch.cat([self.x,self.y,self.z],1)