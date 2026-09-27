"""Our stream order reproduces upstream build_mixed_stream_loader exactly."""
from types import SimpleNamespace

import pytest

from vins.tins_dev import build_order, import_tins

pytestmark = pytest.mark.hades


class _Data:
    def __init__(self, n):
        self.n, self.root = n, "dummy"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        raise AssertionError("never iterated")


class _Loader:
    def __init__(self, n):
        self.dataset = _Data(n)


@pytest.mark.parametrize("seed", [123, 124, 125])
@pytest.mark.parametrize("n_id,n_ood", [(18000, 5000), (18000, 1763), (1800, 500)])
def test_order_matches_upstream(seed, n_id, n_ood):
    t = import_tins()
    args = SimpleNamespace(stream_shuffle=True, stream_seed=seed, batch_size=256)
    loader = t.build_mixed_stream_loader(args, _Loader(n_id), _Loader(n_ood), "dummy")
    assert loader.dataset.order == build_order(n_id, n_ood, seed)
