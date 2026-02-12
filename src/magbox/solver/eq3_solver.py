from magbox.solver.eq_solver import eq_solver
from .vec_normalize import vec_normalize
import torch

class eq3_solver(eq_solver):
    def _after_process(self, y: torch.Tensor) -> torch.Tensor:
        return vec_normalize(y)