from __future__ import annotations
import torch
import scipy.sparse
import numpy as np
from typing import Optional, Callable, cast
from . import boxlib
from .initial import Vars, Lattice
from .spin3 import spin3
from dataclasses import replace

class heff3:
    def __init__(self, sp:spin3, vars:Vars): # B=0,B_dir=[0,0,1],J=1,K_dir=[0,0,1],K1=1,D=0
        # self.spin = spin

        B=vars.B # external field
        B_dir=vars.B_dir # external field direction

        J=vars.J # exchange interaction
        K1_dir=vars.K1_dir # uniaxial anisotropy axis
        K1=vars.K1 # uniaxial anisotropy strength
        D=vars.D # Dzyaloshinskii-Moriya interaction

        self.num=sp.num
        self.B=torch.tensor(B,dtype=sp.dtype,device=sp.device)# external field
        self.B_dir=torch.tensor(B_dir,dtype=sp.dtype,device=sp.device)
        self.B_dir=self.B_dir.view(3)
        self.B_dir=self.B_dir/torch.norm(self.B_dir)
        self.B=self.B*self.B_dir

        self.K_dir=torch.tensor(K1_dir,dtype=sp.dtype,device=sp.device)
        self.K_dir=self.K_dir.view(3)
        self.K_dir=self.K_dir/torch.norm(self.K_dir)
        self.K1=torch.tensor(K1,dtype=sp.dtype,device=sp.device)
        self.K1=self.K1*self.K_dir

        self.H = torch.zeros((self.num,3))
        self.data_type=sp.dtype
        self.device=sp.device

        self.custom_field_fn = vars.custom_heff
        
        raw_kernel = vars.custom_kernel
        if isinstance(raw_kernel, tuple):
            processed_kernel = []
            for item in raw_kernel:
                if isinstance(item, torch.Tensor):
                    processed_kernel.append(item)
                elif scipy.sparse.issparse(item):
                    coo = item.tocoo()
                    values = coo.data
                    indices = np.array([coo.row, coo.col])
                    i = torch.tensor(indices, dtype=torch.long, device=self.device)
                    sparse_values = torch.tensor(values, dtype=self.data_type, device=self.device)
                    t = torch.sparse_coo_tensor(i, sparse_values, coo.shape, device=self.device)
                    processed_kernel.append(t)
                else:
                    t = torch.as_tensor(item, device=self.device,dtype=self.data_type)
                    if t.numel() > 0:
                        sparsity = 1.0 - (float(t.count_nonzero()) / t.numel())
                        if sparsity >= 0.5:
                            t = t.to_sparse()
                    processed_kernel.append(t)
            self.custom_kernel = tuple(processed_kernel)
        else:
            self.custom_kernel = raw_kernel

        if self.custom_field_fn is not None and not callable(self.custom_field_fn):
            raise TypeError("Vars.custom_heff must be callable or None")
        l_type=sp.lattice_type
        N_dim=l_type.size
        N_dim=len(N_dim)

        if isinstance(J,(int,float)) or (isinstance(J,list) and len(J)==1):
            J_val=J[0] if isinstance(J,list) else J
            self.has_J = bool(J_val)
            self.Jmtx=J_val*boxlib.get_Jmtx(replace(l_type,J_direction=None), device=sp.device,data_type=sp.dtype)
        elif isinstance(J,list):
            if len(J)!=N_dim:
                raise ValueError(f"J should be a scalar or a list of length {N_dim}")
            # self.Jmtx=J.to(device=spin.device,dtype=spin.data_type)
            self.Jmtx=torch.zeros((self.num,self.num),dtype=sp.dtype,device=sp.device)
            for i in range(N_dim):
                self.Jmtx+=J[i]*boxlib.get_Jmtx(replace(l_type,J_direction=i), device=sp.device,data_type=sp.dtype)
            self.has_J = any(bool(v) for v in J)

        self.has_B = bool(torch.norm(self.B).item())
        self.has_K1 = bool(torch.norm(self.K1).item())
        self.has_custom = self.custom_field_fn is not None

        if self.has_custom:
            dummy_cartS = torch.ones((self.num,3), dtype=self.data_type, device=self.device)
            dummy_cartS = torch.nn.functional.normalize(dummy_cartS, dim=1,p=2)
            try:
                custom_out = self.custom_field_fn(self.custom_kernel, dummy_cartS) # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError("custom_heff raised an exception during shape validation") from exc

            if not isinstance(custom_out, torch.Tensor):
                raise TypeError("custom_heff must return a torch.Tensor")
            if custom_out.shape != (self.num,3):
                raise ValueError(f"custom_heff output shape should be {(self.num,3)}, got {tuple(custom_out.shape)}")
    def energy(self, S: torch.Tensor) -> torch.Tensor:
        x = S[::3, : ]
        y = S[::3, : ]
        z = S[::3, : ]
        t_num=x.shape[1]

        cartS=self.get_cart_S(x,y,z)

        E_total = torch.zeros(t_num, dtype=self.data_type, device=self.device)

        if self.has_B:
            E_zeeman= - torch.sum(self.B[0] * x + self.B[1] * y + self.B[2] * z, dim=0)
            E_total = E_total + E_zeeman

        if self.has_J:
            E_exch= -0.5 * torch.sum(cartS * (self.Jmtx @ cartS), dim=0)
            E_exch=E_exch[0::3] + E_exch[1::3] + E_exch[2::3]
            E_total = E_total + E_exch

        if self.has_K1:
            E_anis= -0.5 * torch.norm(self.K1) * torch.sum((x *self.K_dir[0] + y * self.K_dir[1] + z *self.K_dir[2])**2, dim=0)
            E_total = E_total + E_anis

        return E_total
    def zeeman3(self) -> torch.Tensor:
        return self.B*torch.ones((self.num,3),dtype=self.data_type,device=self.device)
    # def zeeman2(self,ctheta,stheta,cphi,sphi):
    #     h_theta=self.B_dir@torch.cat([])
    # return self.B*torch.cat([*],1)

    def dipole3(self) -> torch.Tensor:
        # calculate dipole field
        return torch.zeros((self.num,3), dtype=self.data_type, device=self.device)

    def exchange3(self, cartS: torch.Tensor) -> torch.Tensor:
        # calculate exchange field
        return self.Jmtx @ cartS

    def uni_anisotropy3(self, cartS: torch.Tensor) -> torch.Tensor:
        # calculate anisotropy field
        return self.K1 * cartS

    def DM3(self) -> torch.Tensor:
        # calculate DM field
        return torch.zeros((self.num,3), dtype=self.data_type, device=self.device)

    def custom3(self, cartS: torch.Tensor) -> torch.Tensor:
        # user provided effective field
        if self.custom_field_fn is None:
            raise RuntimeError("custom_heff is not set; call `custom3` only when `vars.custom_heff` is provided")
        return self.custom_field_fn(self.custom_kernel, cartS)

    def all3(self, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        cartS=self.get_cart_S(x,y,z)
        h_parts=[]
        if self.has_B:
            h_parts.append(self.zeeman3())
        if self.has_J:
            h_parts.append(self.exchange3(cartS))
        if self.has_K1:
            h_parts.append(self.uni_anisotropy3(cartS))
        if self.has_custom:
            h_parts.append(self.custom3(cartS))

        h0 = torch.zeros((self.num,3), dtype=self.data_type, device=self.device) if not h_parts else torch.stack(h_parts).sum(dim=0)
        return h0.reshape(-1,1)
    
    @staticmethod
    def get_cart_S(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y, z], dim=1)
    
        