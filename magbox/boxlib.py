import torch
def get_data_type(type):
    if type=="f32":
        data_type=torch.float32
    elif type=="f64":
        data_type=torch.float64
    elif type=='f16':
        data_type=torch.float16
    else:
        raise ValueError("type must be f16, f32 or f64")
    return data_type
    