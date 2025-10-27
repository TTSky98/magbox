from dataclasses import dataclass
from typing import Union

@dataclass
class Lattice:
    size: list
    type: str
    periodic: Union[bool , list[bool]] = True
    J_direction: Union[int, None] = None

@dataclass
class Vars:
    B :float=0.0
    B_dir :list[float]= [0.0,0.0,1.0]

    J :Union[float,list[float]] =1.0
    K1_dir :list[float] = [0.0,0.0,1.0]
    K1 :float=1.0

    D :float=0.0