"""Step 1: build the dev splits (no test data is read).

Label space: hold out 100 ImageNet-1K classes (WordNet direct-parent siblings stay in ID, at most one
class per parent, seeded random choice); the other 900 classes are ID.
Splits: support (12/class) + calib (4/class) from TINS's 16-shot list, ID dev (20/class, train, not shots),
near dev (50/class from held-out train), far dev (OpenOOD OpenImage-O val list).
"""
import collections
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .sealing import check_manifest_coverage, load_sealed

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
SMOKE_ID_PER_CLASS = 2
SMOKE_NEAR_PER_CLASS = 5
SMOKE_FAR = 176


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_imagenet_classes(tins_dir=None):
    tins_dir = Path(tins_dir or C.TINS_DIR)
    index = json.loads((tins_dir / "data/ImageNet/imagenet_class_index.json").read_text())
    wnids = [index[str(i)][0] for i in range(1000)]
    raw_names = [index[str(i)][1] for i in range(1000)]
    clean = [str(x) for x in np.load(tins_dir / "data/ImageNet/imagenet_class_clean.npy")]
    assert len(clean) == 1000
    return wnids, raw_names, clean


def wordnet_info(wnids):
    import nltk

    if str(C.NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(C.NLTK_DATA))
    from nltk.corpus import wordnet as wn

    parents, lemmas = {}, {}
    for wnid in wnids:
        synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
        parents[wnid] = sorted({p.name() for p in synset.hypernyms() + synset.instance_hypernyms()})
        lemmas[wnid] = [lemma.name() for lemma in synset.lemmas()]
    return parents, lemmas, str(wn.get_version())


def select_heldout(wnids, clean, parents, seed=None, n=C.N_HELDOUT, exclude=frozenset()):
    """Seeded greedy choice. A chosen class consumes all of its direct parents, so every held-out class
    keeps all of its parent-siblings in ID and no parent contributes more than one held-out class.
    `exclude`: classes that may not be held out (e.g. those held out in a previous dev split)."""
    seed = C.SPLIT_SEED if seed is None else seed
    children = collections.defaultdict(set)
    for wnid in wnids:
        for parent in parents[wnid]:
            children[parent].add(wnid)
    name_count = collections.Counter(clean)
    dup_name = {w for w, name in zip(wnids, clean) if name_count[name] > 1}
    candidates = [
        w for w in wnids
        if w not in dup_name and w not in exclude and any(len(children[p]) >= 2 for p in parents[w])
    ]
    order = list(candidates)
    random.Random(seed).shuffle(order)
    used, held = set(), []
    for wnid in order:
        if set(parents[wnid]) & used:
            continue
        siblings = {c for p in parents[wnid] for c in children[p]} - {wnid}
        if not siblings - set(held):
            continue
        held.append(wnid)
        used.update(parents[wnid])
        if len(held) == n:
            break
    if len(held) != n:
        raise RuntimeError(f"only {len(held)} held-out classes are feasible")
    held_set = set(held)
    info = {}
    for wnid in held:
        siblings = sorted({c for p in parents[wnid] for c in children[p]} - {wnid})
        info[wnid] = {"parents": parents[wnid], "id_siblings": [c for c in siblings if c not in held_set]}
        assert info[wnid]["id_siblings"], wnid
    stats = {
        "n_parents": len(children),
        "n_parents_with_ge2_in1k_children": sum(len(v) >= 2 for v in children.values()),
        "n_candidates": len(candidates),
        "dup_name_excluded": sorted(dup_name),
        "previous_heldout_excluded": sorted(exclude),
        "selection_order": held,
    }
    return held, info, stats


def parse_proto_list(path, wnids):
    by_label = collections.defaultdict(list)
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rel, label = line.split()
                by_label[int(label)].append(rel)
    assert sorted(by_label) == list(range(1000))
    for label, rels in by_label.items():
        assert len(rels) == C.N_SHOT, (label, len(rels))
        names = [Path(r).name for r in rels]
        assert names == sorted(names), f"shot list of class {label} is not lexicographic"
        assert all(Path(r).parts[:2] == ("train", wnids[label]) for r in rels), label
    return by_label


def list_class_files(wnid):
    folder = C.IMAGENET_ROOT / "train" / wnid
    return sorted(e.name for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(IMG_EXTENSIONS))


def sample_files(folder, candidates, k, rng, sealed_paths, sealed_hashes, taken):
    picked, rejected = [], []
    for index in rng.permutation(len(candidates)):
        name = candidates[index]
        real = os.path.realpath(folder / name)
        if real in sealed_paths:
            rejected.append({"file": name, "reason": "sealed_path"})
            continue
        digest = sha256_file(real)
        if digest in sealed_hashes:
            rejected.append({"file": name, "reason": "sealed_hash"})
            continue
        if digest in taken:
            rejected.append({"file": name, "reason": "dev_duplicate"})
            continue
        taken.add(digest)
        picked.append((name, real, digest, os.path.getsize(real)))
        if len(picked) == k:
            break
    if len(picked) < k:
        raise RuntimeError(f"{folder}: only {len(picked)} of {k} eligible images")
    return sorted(picked), rejected


def build(out_dir=None):
    start = time.time()
    out_dir = Path(out_dir or C.SPLITS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    proto_copy = C.INPUTS_DIR / "prototype_train16.txt"
    shutil.copyfile(C.PROTO_LIST_SRC, proto_copy)

    sealed_paths, sealed_hashes, sealed_counts = load_sealed()
    coverage_problems = check_manifest_coverage(sealed_counts)
    if coverage_problems:
        raise RuntimeError(f"seal manifest does not cover the sealed lists: {coverage_problems}")

    wnids, raw_names, clean = load_imagenet_classes()
    parents, lemmas, wn_version = wordnet_info(wnids)
    prev_heldout = set()
    if C.EXCLUDE_HELDOUT_FROM:
        prev_heldout = {h["wnid"] for h in json.loads(Path(C.EXCLUDE_HELDOUT_FROM).read_text())}
    prev_used, prev_hashes = set(), set()          # (wnid, filename) of evaluation images used by another dev
    if C.EXCLUDE_SAMPLES_FROM:
        prev = pd.read_parquet(C.EXCLUDE_SAMPLES_FROM)
        prev = prev[prev.split.isin(["id_dev", "near_dev"])]
        prev_used = {(w, Path(r).name) for w, r in zip(prev.wnid, prev.rel_path)}
        prev_hashes = set(prev.sha256)
    held, held_info, selection_stats = select_heldout(wnids, clean, parents, exclude=frozenset(prev_heldout))
    held_set = set(held)
    idx1000 = {w: i for i, w in enumerate(wnids)}
    id_wnids = [w for w in wnids if w not in held_set]
    id_index = {w: i for i, w in enumerate(id_wnids)}
    proto = parse_proto_list(proto_copy, wnids)
    codex_excluded = {Path(e["path"]).name for e in json.loads(C.PROTO_RESOLUTION_SRC.read_text())["excluded"]}

    rows, taken, rejections = [], set(), {"id_dev": {}, "near_dev": {}, "far_dev": []}

    def add(split, group, wnid, rel, real, digest, nbytes, smoke, parents_=None, siblings=None):
        is_imagenet = wnid is not None
        rows.append(
            {
                "sample_id": f"in1k/{rel}" if is_imagenet else f"oio/{rel}",
                "split": split,
                "group": group,
                "wnid": wnid or "",
                "class_idx_1k": idx1000[wnid] if is_imagenet else -1,
                "class_idx_id": id_index.get(wnid, -1) if is_imagenet else -1,
                "class_name": clean[idx1000[wnid]] if is_imagenet else "",
                "rel_path": rel,
                "path": real,
                "sha256": digest,
                "bytes": int(nbytes),
                "smoke": bool(smoke),
                "near_parents": json.dumps(parents_ or []),
                "near_id_siblings": json.dumps(siblings or []),
            }
        )

    # support / calib: TINS 16-shot list, first 12 lines -> support, last 4 -> calib.
    # ImageNet has a few byte-identical images filed under two classes; TINS's list keeps them. We keep them
    # too (upstream prototypes are unchanged) as long as every copy is in the support split of different
    # classes; any duplicate touching calib is an error. All shot hashes are excluded from the dev splits.
    shot_owner, support_dups = {}, []
    for wnid in id_wnids:
        for j, rel in enumerate(proto[idx1000[wnid]]):
            split = "support" if j < C.N_SUPPORT else "calib"
            real = os.path.realpath(C.IMAGENET_ROOT / rel)
            digest = sha256_file(real)
            if real in sealed_paths or digest in sealed_hashes:
                raise RuntimeError(f"shot image is sealed: {real}")
            if digest in shot_owner:
                prev_split, prev_wnid, prev_rel = shot_owner[digest]
                if not (split == prev_split == "support" and prev_wnid != wnid):
                    raise RuntimeError(f"duplicate shot image touching calib or within a class: {real}")
                support_dups.append({"sha256": digest, "copies": [prev_rel, rel]})
            else:
                shot_owner[digest] = (split, wnid, rel)
            taken.add(digest)
            add(split, "ID", wnid, rel, real, digest, os.path.getsize(real), True)

    # evaluation images of a previous dev split are never reused (by file and by content)
    taken |= prev_hashes

    # ID dev: 20 per ID class from train, excluding all 16 shots (and Codex's replaced duplicates)
    for wnid in id_wnids:
        shots = {Path(r).name for r in proto[idx1000[wnid]]}
        candidates = [f for f in list_class_files(wnid)
                      if f not in shots and f not in codex_excluded and (wnid, f) not in prev_used]
        rng = np.random.default_rng([C.SPLIT_SEED, 1, idx1000[wnid]])
        picked, rejected = sample_files(C.IMAGENET_ROOT / "train" / wnid, candidates, C.N_ID_DEV_PER_CLASS,
                                        rng, sealed_paths, sealed_hashes, taken)
        if rejected:
            rejections["id_dev"][wnid] = rejected
        for j, (name, real, digest, nbytes) in enumerate(picked):
            add("id_dev", "ID", wnid, f"train/{wnid}/{name}", real, digest, nbytes, j < SMOKE_ID_PER_CLASS)

    # near dev: 50 per held-out class from train
    for wnid in sorted(held, key=idx1000.get):
        rng = np.random.default_rng([C.SPLIT_SEED, 2, idx1000[wnid]])
        candidates = [f for f in list_class_files(wnid) if (wnid, f) not in prev_used]
        picked, rejected = sample_files(C.IMAGENET_ROOT / "train" / wnid, candidates,
                                        C.N_NEAR_PER_CLASS, rng, sealed_paths, sealed_hashes, taken)
        if rejected:
            rejections["near_dev"][wnid] = rejected
        for j, (name, real, digest, nbytes) in enumerate(picked):
            add("near_dev", "near", wnid, f"train/{wnid}/{name}", real, digest, nbytes,
                j < SMOKE_NEAR_PER_CLASS, held_info[wnid]["parents"], held_info[wnid]["id_siblings"])

    # far dev: OpenOOD OpenImage-O val split
    n_far = 0
    with open(C.FAR_DEV_LIST) as handle:
        for line in handle:
            if not line.strip():
                continue
            rel = line.split()[0]
            real = os.path.realpath(C.OPENOOD_IMAGES / rel)
            if real in sealed_paths:
                rejections["far_dev"].append({"file": rel, "reason": "sealed_path"})
                continue
            digest = sha256_file(real)
            if digest in sealed_hashes or digest in taken:
                reason = "sealed_hash" if digest in sealed_hashes else "dev_duplicate"
                rejections["far_dev"].append({"file": rel, "reason": reason})
                continue
            taken.add(digest)
            add("far_dev", "far", None, rel, real, digest, os.path.getsize(real), n_far < SMOKE_FAR)
            n_far += 1

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "samples.parquet", index=False)

    heldout = []
    for wnid in sorted(held, key=idx1000.get):
        heldout.append(
            {
                "wnid": wnid,
                "idx_1k": idx1000[wnid],
                "clean_name": clean[idx1000[wnid]],
                "raw_name": raw_names[idx1000[wnid]],
                "lemmas": lemmas[wnid],
                "parents": held_info[wnid]["parents"],
                "id_siblings": [{"wnid": s, "clean_name": clean[idx1000[s]]} for s in held_info[wnid]["id_siblings"]],
            }
        )
    (out_dir / "heldout.json").write_text(json.dumps(heldout, indent=1) + "\n")
    id_classes = [{"id_idx": i, "wnid": w, "idx_1k": idx1000[w], "clean_name": clean[idx1000[w]]}
                  for i, w in enumerate(id_wnids)]
    (out_dir / "id_classes.json").write_text(json.dumps(id_classes, indent=1) + "\n")
    info = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_seed": C.SPLIT_SEED,
        "counts": frame.groupby("split").size().to_dict(),
        "smoke_counts": frame[frame.smoke].groupby("split").size().to_dict(),
        "rejections": rejections,
        "n_rejections": {
            "id_dev": sum(len(v) for v in rejections["id_dev"].values()),
            "near_dev": sum(len(v) for v in rejections["near_dev"].values()),
            "far_dev": len(rejections["far_dev"]),
        },
        "support_cross_class_duplicates": support_dups,
        "previous_dev_exclusion": {"heldout_from": C.EXCLUDE_HELDOUT_FROM, "samples_from": C.EXCLUDE_SAMPLES_FROM,
                                   "n_classes_excluded_from_heldout": len(prev_heldout),
                                   "n_images_excluded": len(prev_used)},
        "sealed_manifest_counts": sealed_counts,
        "wordnet_version": wn_version,
        "heldout_selection": selection_stats,
        "input_sha256": {
            "prototype_train16.txt": sha256_file(proto_copy),
            "val_openimage_o.txt": sha256_file(C.FAR_DEV_LIST),
            "seal_manifest.jsonl": sha256_file(C.SEAL_MANIFEST),
        },
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return frame, info


if __name__ == "__main__":
    _, summary = build()
    print(json.dumps({k: summary[k] for k in ("counts", "smoke_counts", "n_rejections", "seconds")}, indent=1))
