"""Test-split sealing: no dev image is (or duplicates) a sealed image, and sealed paths cannot be opened."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from vins import config as C
from vins.sealing import Guard, SealedAccessError, check_manifest_coverage, load_sealed

pytestmark = pytest.mark.hades
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sealed():
    return load_sealed()


@pytest.fixture(scope="module")
def samples():
    return pd.read_parquet(C.SPLITS_DIR / "samples.parquet")


def test_manifest_covers_every_sealed_list(sealed):
    _, _, counts = sealed
    assert not check_manifest_coverage(counts)
    assert sum(counts.values()) == 130908 + 5000


def test_no_dev_path_is_sealed(sealed, samples):
    paths, _, _ = sealed
    assert not set(samples.path) & paths


def test_no_dev_image_is_byte_identical_to_a_sealed_image(sealed, samples):
    _, hashes, _ = sealed
    assert not set(samples.sha256) & hashes


def test_guard_refuses_sealed_paths(sealed, samples):
    paths, _, _ = sealed
    guard = Guard(paths)
    with open(C.SEAL_MANIFEST) as handle:
        sealed_path = json.loads(handle.readline())["path"]
    with pytest.raises(SealedAccessError):
        guard.check(sealed_path)
    assert guard.check(samples.path.iloc[0]) == samples.path.iloc[0]


def test_sealed_list_names_only_in_config_and_sealing():
    names = [re.escape(n) for n in C.SEALED_LIST_FILES]
    pattern = re.compile("|".join(names))
    offenders = []
    for path in list((ROOT / "vins").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if path.name in {"config.py", "sealing.py"}:
            continue
        if pattern.search(path.read_text()):
            offenders.append(path.name)
    assert not offenders


def test_features_cover_only_dev_samples(samples):
    import torch

    for feat in C.FEATURES:
        path = C.FEATURES_DIR / f"{feat}.pt"
        if not path.exists():
            pytest.skip("features not computed yet")
        blob = torch.load(path, map_location="cpu")
        assert blob["sample_id"] == samples.sample_id.tolist()
