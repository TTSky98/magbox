import torch
from magbox.boxlib import get_Jmtx
import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP.*")

lt={"type":"square","size":[8,6,4],"periodic":True,"direction":None}
tmp=get_Jmtx(lt)
print(tmp)
plt.figure(figsize=(6,6))
plt.spy(tmp.to_dense().cpu())
plt.show()

# def test_get_Jmtx():
#     ...
# def test_ode_rk45(