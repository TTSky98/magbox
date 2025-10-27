from dataclasses import dataclass, field
from typing import Union

@dataclass
class Lattice:
    size: list = field(default_factory=list)
    type: str = field(default_factory=str)
    periodic: Union[bool, list[bool]] = field(default_factory=lambda: True)
    J_direction: Union[int, None] = field(default_factory=lambda: None)

@dataclass
class Vars:
    B: float = field(default_factory=lambda: 0.0)
    B_dir: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    J: Union[float, list[float]] = field(default_factory=lambda: 1.0)
    K1_dir: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    K1: float = field(default_factory=lambda: 1.0)
    D: float = field(default_factory=lambda: 0.0)
