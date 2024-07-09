import pytest

import torch

from repro import o3
from e3nn.util.test import assert_equivariant, assert_auto_jitable, assert_no_graph_break, random_irreps


@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4))
@pytest.mark.parametrize("squared", [True, False])
def test_norm_no_graph_break(irreps_in, squared) -> None:
    """Check whether norm compiles without graph breaks"""

    mod = o3.Norm(irreps_in, squared=squared)
    x = torch.randn(mod.irreps_in.dim)
    torch._logging.set_logs(graph_breaks=True, bytecode=True, recompiles=True, graph=True)
    torch.compile(mod)
