from magbox.solver.sde_solver import sde_solver
from .vec_normalize import vec_normalize
import torch

class sde3_solver(sde_solver):
    def _after_process(self, y: torch.Tensor) -> torch.Tensor:
        return vec_normalize(y)