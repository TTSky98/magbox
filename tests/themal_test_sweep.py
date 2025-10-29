import numpy as np 
from thermal_tools import run_thermal, plot_thermal
from pathlib import Path

# Temp_all=np.logspace(np.log10(0.1),np.log10(10),4)
# dt_all=[0.01,0.1,0.5,1.0]
# alpha_all=[0.01,0.1,0.2,0.3]

# gamma_all=[0.5,1.0,2.0]

# for i,Temp in enumerate(Temp_all):
#     for j,dt in enumerate(dt_all):
#         for k,alpha in enumerate(alpha_all):
#             for l,gamma in enumerate(gamma_all):
#                 id=i*1000+j*100+k*10+l+1
#                 print(f"Running test id={id} with Temp={Temp}, dt={dt}, alpha={alpha}, gamma={gamma}")
                
#                 file=run_thermal(dt=dt, alpha=alpha, Temp=Temp,
#                             gamma=gamma, T=400, run_id=id)
#                 plot_thermal(file, start_time=40, bins=80)
#                 print(f"Completed test id={id} with Temp={Temp}, dt={dt}, alpha={alpha}, gamma={gamma}\n")

file=run_thermal(dt=0.1, alpha=0.1, Temp=0.05, T=200, dtype='f64', device='cpu', spin_num=2**8)

file=Path("./thermal_test_1.npz")
plot_thermal(file, start_time=50, bins=80,fit_counts=60)