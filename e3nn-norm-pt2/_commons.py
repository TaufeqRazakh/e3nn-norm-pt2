
from typing import NamedTuple

class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple

def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out

