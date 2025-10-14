import numpy as np
import torch
from magbox import boxlib
class spin:
    def __init__(self, theta, phi, lattice_type,type="f32"):
        data_type=boxlib.get_data_type(type)
        l_theta=len(theta)
        l_phi=len(phi)
        if l_theta!=l_phi:
            raise ValueError("theta and phi must have the same length")
        
        self.num = l_theta
        self.theta = torch.tensor(theta,dtype=data_type)
        self.phi = torch.tensor(phi,dtype=data_type)
        self.theta=self.theta.view(-1,1)
        self.phi=self.phi.view(-1,1)
        self.type=lattice_type
    def cart(self): 
        # convert to cartesian coordinates
        s_theta=np.sin(self.theta) 
        x = s_theta * np.cos(self.phi)
        y = s_theta * np.sin(self.phi)
        z = np.cos(self.theta)
        return x, y, z