from magbox import llg,spin, Lattice, Vars
from torch import sin, cos
from matplotlib import pyplot as plt
import numpy as np
import cProfile, pstats, io
# from pyinstrument import Profiler

pr=cProfile.Profile()
pr.enable()

# trace(files=files.all)

# pr=Profiler()
# pr.start()

rng=np.random.default_rng()

N1=128
N2=128
lt = Lattice(type="square", size=[N1, N2], periodic=True)
vars = Vars(J=0)
theta0=np.acos(rng.random([N1,N2])*2-1)
# theta0=np.ones([N1])*0.1
phi0=rng.random([N1,N2])*2*np.pi
# theta0=np.ones([N1])*0.1
# phi0=np.zeros([N1])
alpha=0.1


sp=spin(theta0,phi0,lattice_type=lt,device="cpu",type='f64')
sf=llg(sp,vars=vars, dt=0.1,alpha=alpha,T=1,Temp=0.1)
ang=sp.ang

t,ang,stats,erro_info=sf.run(sp)

en=sf.h_fun.energy(ang)

t_np=t.cpu().detach().numpy()
en_np=en.cpu().detach().numpy()
en_np=en_np/N1

pr.disable()
s = io.StringIO()
sortby = "cumtime"  
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# print(s.getvalue())
pr.dump_stats("llg_thermal_cpu_f64.prof")

# pr.stop()
# pr.open_in_browser()

plt.figure(figsize=(6,6))

plt.subplot(2,1,1)
plt.plot(t_np,en_np,label="energy")
plt.xlabel("Time")
plt.title("energy")
plt.legend()

plt.tight_layout()
plt.show()


