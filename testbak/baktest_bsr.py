import torch

crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 2])
values1 = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                       [[3, 4, 5], [9, 10, 11]],
                       [[12, 13, 14], [18, 19, 20]],
                       [[15, 16, 17], [21, 22, 23]]])
values2 = torch.tensor([[[0, 1,], [6, 7]],
                       [[3, 4], [9, 10]],
                       [[12, 13], [18, 19]],
                       [[15, 16], [21, 22]]])
bsr1 = torch.sparse_bsr_tensor(crow_indices, col_indices, values1, dtype=torch.float64,device=torch.device("cuda"))
bsr2= torch.sparse_bsr_tensor(crow_indices, col_indices, values2, dtype=torch.float64,device=torch.device("cpu"))
v1=torch.tensor([[1],[2],[3],[4],[5],[6],[4],[5],[6]],dtype=torch.float64,device=torch.device("cuda"))
v2=torch.tensor([[1],[2],[3],[4],[5],[6]],dtype=torch.float64,device=torch.device("cpu"))
# print(bsr1 @ v1)
print(bsr2 @ v2)