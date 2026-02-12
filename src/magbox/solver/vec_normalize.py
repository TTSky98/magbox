import torch

def vec_normalize(y:torch.Tensor):
    if y.shape[0] == 0: 
        return y
    else:
        return torch.nn.functional.normalize(y.view(-1,3), dim=1,p=2).view_as(y)