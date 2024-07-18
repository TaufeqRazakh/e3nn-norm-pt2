from graphnorm import Norm, assert_no_graph_break

import pytest

import torch


@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"])
@pytest.mark.parametrize("squared", [True, False])
def test_norm_no_graph_break(irreps_in, squared) -> None:
    """Check whether norm compiles without graph breaks"""

    mod = Norm(irreps_in, squared=squared)
    x = torch.randn(mod.irreps_in.dim)
    torch._logging.set_logs(graph_breaks=True, bytecode=True, recompiles=True, graph=True)
    assert_no_graph_break(mod,x)
    # torch.compile(mod)