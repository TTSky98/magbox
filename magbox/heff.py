import torch

class heff:
    def __init__(self, spin, B):
        self.spin = spin
        l=spin.num
        self.B=B # external field
        self.H = torch.zeros((l,3))
    def zeeman(self):
        return self.B
    
    def dipole(self):
        # calculate dipole field
        pass
    def exchange(self):
        # calculate exchange field
        pass
    def uni_anisotropy(self):
        # calculate anisotropy field
        pass
    def DM(self):
        # calculate DM field
        pass
    def all(self):
        return self.zeeman() + self.dipole() + self.exchange() + self.uni_anisotropy() + self.DM()