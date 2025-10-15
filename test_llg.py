from magbox import llg,spin
import torch
from matplotlib import pyplot as plt

lt={"type":"square","size":[5,5],"periodic":True}
sp=spin(0.1*torch.ones([5,5]),torch.zeros([5,5]),lattice_type=lt,device="gpu")
sf=llg(sp)

kernal=sf.llg_drift(sp)
plt.figure(figsize=(6,6))
plt.spy(kernal.to_dense().cpu())
plt.show()


