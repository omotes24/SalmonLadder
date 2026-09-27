"""Step 1 splits: disjointness, sizes, held-out rules, shot handling (runs on hades after build)."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from vins import config as C

pytestmark = pytest.mark.hades


@pytest.fixture(scope="module")
def samples():
    return pd.read_parquet(C.SPLITS_DIR / "samples.parquet")


@pytest.fixture(scope="module")
def heldout():
    return json.loads((C.SPLITS_DIR / "heldout.json").read_text())


def test_ids_paths_and_hashes_disjoint(samples):
    assert samples.sample_id.is_unique
    assert samples.path.is_unique
    # no image content is shared between two splits
    assert (samples.groupby("sha256").split.nunique() == 1).all()
    # the only repeated content: cross-class duplicates inside TINS's shot list, all in support
    dup = samples[samples.sha256.duplicated(keep=False)]
    assert set(dup.split) <= {"support"}
    assert (dup.groupby("sha256").wnid.nunique() == dup.groupby("sha256").size()).all()
    info = json.loads((C.SPLITS_DIR / "build_info.json").read_text())
    assert len(info["support_cross_class_duplicates"]) == len(dup) - dup.sha256.nunique()
    rest = samples[samples.split != "support"]
    assert rest.sha256.is_unique and not set(rest.sha256) & set(samples[samples.split == "support"].sha256)


def test_split_sizes(samples):
    counts = samples.split.value_counts().to_dict()
    info = json.loads((C.SPLITS_DIR / "build_info.json").read_text())
    assert counts["support"] == 900 * C.N_SUPPORT
    assert counts["calib"] == 900 * C.N_CALIB
    assert counts["id_dev"] == 900 * C.N_ID_DEV_PER_CLASS
    assert counts["near_dev"] == C.N_HELDOUT * C.N_NEAR_PER_CLASS
    assert counts["far_dev"] == 1763 - info["n_rejections"]["far_dev"]
    per_class = samples[samples.split == "id_dev"].groupby("wnid").size()
    assert len(per_class) == 900 and (per_class == C.N_ID_DEV_PER_CLASS).all()
    per_held = samples[samples.split == "near_dev"].groupby("wnid").size()
    assert len(per_held) == C.N_HELDOUT and (per_held == C.N_NEAR_PER_CLASS).all()


def test_heldout_rules(samples, heldout):
    from vins.splits import load_imagenet_classes, wordnet_info

    wnids, _, clean = load_imagenet_classes()
    parents, _, _ = wordnet_info(wnids)
    held = [h["wnid"] for h in heldout]
    assert len(held) == len(set(held)) == C.N_HELDOUT
    used = Counter(p for w in held for p in parents[w])
    assert max(used.values()) == 1                      # at most one held-out class per parent
    names = Counter(clean)
    assert all(names[clean[wnids.index(w)]] == 1 for w in held)   # duplicate-name classes excluded
    id_set = set(wnids) - set(held)
    for h in heldout:
        sib = [s["wnid"] for s in h["id_siblings"]]
        assert sib and set(sib) <= id_set
        assert all(set(parents[s]) & set(parents[h["wnid"]]) for s in sib)
    assert set(samples[samples.group == "ID"].wnid) == id_set
    assert set(samples[samples.group == "near"].wnid) == set(held)
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    assert [c["wnid"] for c in id_classes] == [w for w in wnids if w in id_set]


def test_support_calib_follow_tins_shot_list(samples):
    lines = [l.split() for l in (C.INPUTS_DIR / "prototype_train16.txt").read_text().splitlines() if l.strip()]
    by_label = {}
    for rel, label in lines:
        by_label.setdefault(int(label), []).append(rel)
    for (idx, split), group in samples[samples.split.isin(["support", "calib"])].groupby(["class_idx_1k", "split"]):
        expected = by_label[idx][:C.N_SUPPORT] if split == "support" else by_label[idx][C.N_SUPPORT:]
        assert group.rel_path.tolist() == expected


def test_id_dev_excludes_all_shots(samples):
    shots = {l.split()[0] for l in (C.INPUTS_DIR / "prototype_train16.txt").read_text().splitlines() if l.strip()}
    assert not set(samples[samples.split == "id_dev"].rel_path) & shots


def test_far_dev_from_openimage_o_val(samples):
    listed = {l.split()[0] for l in C.FAR_DEV_LIST.read_text().splitlines() if l.strip()}
    assert set(samples[samples.split == "far_dev"].rel_path) <= listed


def test_recorded_hashes_are_correct(samples):
    from vins.splits import sha256_file

    for row in samples.sample(n=200, random_state=0).itertuples():
        assert sha256_file(row.path) == row.sha256


def test_disjoint_from_previous_dev(samples, heldout):
    if not C.EXCLUDE_SAMPLES_FROM:
        pytest.skip("not a confirmation split")
    prev = pd.read_parquet(C.EXCLUDE_SAMPLES_FROM)
    prev_eval = prev[prev.split.isin(["id_dev", "near_dev"])]
    cur_eval = samples[samples.split.isin(["id_dev", "near_dev"])]
    assert not set(cur_eval.path) & set(prev_eval.path)
    assert not set(cur_eval.sha256) & set(prev_eval.sha256)
    prev_held = {h["wnid"] for h in json.loads(Path(C.EXCLUDE_HELDOUT_FROM).read_text())}
    assert not prev_held & {h["wnid"] for h in heldout}


def test_smoke_subset(samples):
    smoke = samples[samples.smoke]
    counts = smoke.split.value_counts().to_dict()
    assert counts["id_dev"] == 900 * 2 and counts["near_dev"] == C.N_HELDOUT * 5 and counts["far_dev"] == 176
    assert Path(C.SPLITS_DIR / "heldout.json").exists()
