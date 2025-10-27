from magbox import llg,spin, Lattice
from torch import sin, cos
from matplotlib import pyplot as plt
import numpy as np
import cProfile, pstats, io

pr=cProfile.Profile()
rng=np.random.default_rng()

N1=32
N2=32
lt=Lattice(type="square",size=[N1,N2],periodic=True)
theta0=np.acos(rng.random([N1,N2])*2-1)
phi0=rng.random([N1,N2])*2*np.pi
# theta0=np.ones([N1])*0.1
# phi0=np.zeros([N1])
alpha=0.1

pr.enable()
sp=spin(theta0,phi0,lattice_type=lt,device="gpu",type="f32")
sf=llg(sp,dt=1,alpha=alpha,T=200)
ang=sp.ang

t,ang,stats,erro_info=sf.run(sp)
theta=ang[0::2]
phi=ang[1::2]
x=sin(theta)*cos(phi)
y=sin(theta)*sin(phi)
z=cos(theta)

t_np=t.cpu().detach().numpy()
x_np=x.cpu().detach().numpy()
y_np=y.cpu().detach().numpy()
z_np=z.cpu().detach().numpy()
z0=z_np[0,0]
zt=1+np.exp(-2*alpha*t_np)*(z0-1)
pr.disable()

s = io.StringIO()
sortby = "cumtime"  
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# print(s.getvalue())
pr.dump_stats("llg_gpu_f32_no_lambda.prof")

plt.figure(figsize=(6,6))

plt.subplot(2,1,1)
plt.plot(t_np,x_np[0,:],label="m_x")
plt.plot(t_np,y_np[0,:],label="m_y")
plt.xlabel("Time")
plt.title("m_x and m_y")
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_np,z_np[0,:],label="m_z")
# plt.plot(t_np,zt,label="m_z")
plt.xlabel("Time")
plt.title("m_z")
plt.legend()

plt.tight_layout()
plt.show()


