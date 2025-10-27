import numpy as np
import torch
from . import boxlib
from .initial import Lattice

class spin:
    def __init__(self, theta, phi, lattice_type: Lattice,type="f32", device="gpu",thread:int=4):
        torch.set_num_threads(thread)
        data_type=boxlib.get_data_type(type)
        self.data_type=data_type
        self.device=boxlib.get_device(device)
        if isinstance(theta,torch.Tensor):
            self.theta=theta.to(device=self.device,dtype=self.data_type)
        else:
            self.theta = torch.tensor(theta,dtype=self.data_type,device=self.device)

        if isinstance(phi,torch.Tensor):
            self.phi=phi.to(device=self.device,dtype=self.data_type)
        else:
            self.phi = torch.tensor(phi,dtype=self.data_type,device=self.device)

        self.theta=self.theta.view(-1,1)
        self.phi=self.phi.view(-1,1)
        l_theta=len(self.theta)
        l_phi=len(self.phi)
        if l_theta!=l_phi:
            raise ValueError("theta and phi must have the same length")
        self.num = l_theta

        self.ang=torch.cat([self.theta,self.phi],1)
        self.ang=self.ang.reshape(-1,1)
        self.c_theta=torch.cos(self.theta)
        self.s_theta=torch.sin(self.theta)
        self.c_phi=torch.cos(self.phi)
        self.s_phi=torch.sin(self.phi)

        self.x, self.y, self.z = self.cart()
        self.cart_S=torch.cat([self.x,self.y,self.z],1)
        self.lattice_type=lattice_type

        num=lattice_type.size
        if self.num != np.prod(num):
            raise ValueError(f"initial condition theta and phi length {self.num} does not match lattice size {num}")
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

