import magbox
import torch
import time
import numpy as np

if __name__=="__main__":
    N=2**6
    loop=1000
    data_type="f32"
    device_type="gpu"

    dtype=magbox.boxlib.get_data_type(data_type)
    device=magbox.boxlib.get_device(device_type)

    theta0 = np.random.random((N,1)) * 0.1
    phi0 = np.random.random((N,1)) * 2 * np.pi

    x0=np.sin(theta0)*np.cos(phi0)
    y0=np.sin(theta0)*np.sin(phi0)
    z0=np.cos(theta0)

    x=torch.as_tensor(x0,dtype=dtype,device=device)
    y=torch.as_tensor(y0,dtype=dtype,device=device)
    z=torch.as_tensor(z0,dtype=dtype,device=device)

    h=torch.randn(N,3,dtype=dtype,device=device)
    heff=h.view(-1,1)

    lt=magbox.Lattice(size=[N], type="square")
    vars=magbox.Vars()

    sp=magbox.spin3(x,y,z, lattice_type=lt, dtype=data_type, device=device_type, thread=4)
    sf=magbox.llg3(sp,vars)

    S=torch.cat([x,y,z],dim=1)

    t0=time.time()
    for i in range(loop):
        Mmtx =sf.M_cross_mat(x,y,z)
        M_csr=Mmtx @ heff
        M2_csr=Mmtx @ M_csr
    t1=time.time()
    print(f"Sparse matrix multiplication mean time: {t1-t0:.4f} s")

    t0=time.time()
    for i in range(loop):
        M_cross= torch.linalg.cross(S,h,dim=1)
        M2_cross= torch.linalg.cross(S, M_cross,dim=1)
    t1=time.time()
    print(f"Direct cross product mean time: {t1-t0:.4f} s")

    t0=time.time()
    for i in range(loop):
        M_cross_BAC= torch.linalg.cross(S,h,dim=1)
        M2_cross_BAC= S * (torch.linalg.vecdot(S,h)).view(-1,1) - h
    t1=time.time()
    print(f"Direct cross product mean time: {t1-t0:.4f} s")


    M_cross = M_cross.view(-1,1)
    M2_cross = M2_cross.view(-1,1)
    M2_cross_BAC = M2_cross_BAC.view(-1,1)

    diff1=torch.sum(M_csr-M_cross)
    diff1=diff1.item()

    diff2=torch.sum(M2_csr - M2_cross)
    diff2=diff2.item()

    diff3=torch.sum(M2_cross_BAC - M2_cross)
    diff3=diff3.item()
    print(f"Difference 1 between sparse matrix multiplication and direct cross product: {diff1:e}")
    print(f"Difference 2 between sparse matrix multiplication and direct cross product: {diff2:e}")
    print(f"Difference 3 between two direct cross product methods: {diff3:e}")




