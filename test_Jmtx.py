import torch
from magbox.boxlib import get_Jmtx
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP.*")

lt={"type":"square","size":[6,4]}
tmp=get_Jmtx(lt,device=torch.device("cpu"))
print(tmp)
plt.figure(figsize=(6,6))
# plt.spy(tmp.to_dense())
plt.show()

# def test_get_Jmtx():
#     ...
# def test_ode_rk45(