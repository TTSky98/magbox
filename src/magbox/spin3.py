import numpy as np
import torch
from . import boxlib
from .initial import Lattice

class spin3:
    def __init__(self, x, y, z, lattice_type: Lattice,dtype="f32", device="gpu",thread:int=4):
        torch.set_num_threads(thread)

        dtype=boxlib.get_data_type(dtype)
        self.dtype=dtype
        self.device=boxlib.get_device(device)

        self.x = torch.as_tensor(x,dtype=self.dtype,device=self.device)
        self.y = torch.as_tensor(y,dtype=self.dtype,device=self.device)
        self.z = torch.as_tensor(z,dtype=self.dtype,device=self.device)


        self.x=self.x.view(-1,1)
        self.y=self.y.view(-1,1)
        self.z=self.z.view(-1,1)
        l_x=len(self.x)
        l_y=len(self.y)
        l_z=len(self.z)
        assert l_x==l_y and l_y==l_z, "mx, my and mz must have the same length"
        self.num = l_x

        self.update()

        self.lattice_type=lattice_type

        num=lattice_type.size
        assert self.num == np.prod(num), f"initial condition length {self.num} does not match lattice size {num}"
    def update(self): # 根据x,y,z的值重新归一化并更新cart_S的值
        self.cart_S=torch.cat([self.x,self.y,self.z], dim = 1)
        self.cart_S=torch.nn.functional.normalize(self.cart_S, dim=1,p=2)
        self.x, self.y ,self.z = spin3.get_xyz(self.cart_S)
        self.S=self.cart_S.view(-1,1)
    @staticmethod
    def get_xyz(S):
        return S[::3],S[1::3],S[2::3]
        
