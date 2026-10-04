"""Phase 5: build the unused bank U4 (after the lock; no image is scored here, no sealed image is opened).

Five new ImageNet-1K class splits (100 held-out classes each; every held-out class keeps a WordNet sibling in ID; at
most two held-out classes per parent; seeds 20261003..20261007), disjoint from the dev1/dev2 held-out classes;
overlap with the Phase 4 (U1) held-out classes is allowed and reported. Images never used in any earlier phase:
dev1/dev2 samples, the TINS 16-shot list and its replaced duplicates, the five R5 draws, the whole Phase 4 pool and
sealed images are excluded by file name and sha256. Per class: 48 shots (3 draws x (12 support + 4 calibration)),
20 ID-evaluation images, and 50 OOD-evaluation images for classes held out in a U4 split.
"""
import collections
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import GONOGO, IMAGENET, R5, dump, sha_file, utc
from u4_common import BANKS, N_DRAWS, N_HELD, N_IDEVAL, N_OODEVAL, N_SHOT, N_SPARE, N_SPLITS, P4, SEED, require_lock
from vins import config as C
from vins.sealing import load_sealed
from vins.splits import list_class_files, load_imagenet_classes, wordnet_info

G = {}


def select_relaxed(wnids, clean, parents, seed, n, exclude, max_per_parent=2):
    """The Phase 4 rule (build_banks.select_relaxed), unchanged."""
    import random
    children = collections.defaultdict(set)
    for w in wnids:
        for p in parents[w]:
            children[p].add(w)
    count = collections.Counter(clean)
    dup = {w for w, nm in zip(wnids, clean) if count[nm] > 1}
    cand = [w for w in wnids if w not in dup and w not in exclude and any(len(children[p]) >= 2 for p in parents[w])]
    order = list(cand)
    random.Random(seed).shuffle(order)
    held, hs, use = [], set(), collections.Counter()

    def sib(h, heldset):
        return {c for p in parents[h] for c in children[p]} - {h} - heldset

    for w in order:
        if any(use[p] >= max_per_parent for p in parents[w]):
            continue
        new = hs | {w}
        if not sib(w, new):
            continue
        if any(not sib(h, new) for h in held if set(parents[h]) & set(parents[w])):
            continue
        held.append(w)
        hs = new
        use.update(parents[w])
        if len(held) == n:
            break
    if len(held) < n:
        raise RuntimeError(f"relaxed rule: only {len(held)} held-out classes feasible (seed {seed})")
    info = {w: {"parents": parents[w], "id_siblings": sorted(sib(w, hs))} for w in held}
    return held, info, {"rule": f"ID sibling kept, <= {max_per_parent} held-out per parent", "n_candidates": len(cand)}


def used_sets():
    names, hashes = collections.defaultdict(set), set()
    for work in (GONOGO, GONOGO / "dev2"):
        s = pd.read_parquet(work / "splits" / "samples.parquet")
        hashes |= set(s.sha256)
        for w, r in zip(s.wnid, s.rel_path):
            if w:
                names[w].add(Path(r).name)
    for line in (GONOGO / "inputs" / "prototype_train16.txt").read_text().splitlines():
        if line.strip():
            rel = Path(line.split()[0])
            names[rel.parts[1]].add(rel.name)
    for e in json.loads(C.PROTO_RESOLUTION_SRC.read_text())["excluded"]:
        p = Path(e["path"])
        names[p.parent.name].add(p.name)
    r5 = pd.read_parquet(R5 / "shots" / "draws.parquet")
    hashes |= set(r5.sha256)
    for w, r in zip(r5.wnid, r5.rel_path):
        names[w].add(Path(r).name)
    p4 = pd.read_parquet(P4 / "banks" / "imagenet_pool.parquet")
    hashes |= set(p4.sha256)
    for w, p in zip(p4.wnid, p4.path):
        names[w].add(Path(p).name)
    return names, hashes, len(p4)


def pick_class(task):
    idx, wnid, n_need = task
    cand = [f for f in list_class_files(wnid) if f not in G["names"].get(wnid, set())]
    rng = np.random.default_rng([SEED, 1, idx])
    picked, rej, seen = [], collections.Counter(), set()
    for j in rng.permutation(len(cand)):
        real = os.path.realpath(IMAGENET / "train" / wnid / cand[j])
        if real in G["sealed_paths"]:
            rej["sealed_path"] += 1
            continue
        dg = sha_file(real)
        if dg in G["sealed_hashes"]:
            rej["sealed_hash"] += 1
        elif dg in G["used_hashes"]:
            rej["used_hash"] += 1
        elif dg in seen:
            rej["duplicate"] += 1
        else:
            seen.add(dg)
            picked.append((cand[j], real, dg))
            if len(picked) == n_need + N_SPARE:
                break
    if len(picked) < n_need:
        raise RuntimeError(f"{wnid}: {len(picked)} < {n_need} eligible")
    return idx, wnid, picked, dict(rej), len(cand)


def main():
    lock = require_lock()
    out = BANKS
    out.mkdir(parents=True, exist_ok=True)
    if (out / "imagenet_pool.parquet").exists():
        raise SystemExit("U4 already built; refusing to overwrite")
    wnids, raw, clean = load_imagenet_classes()
    parents, lemmas, wn_version = wordnet_info(wnids)
    prev = set()
    for work in (GONOGO, GONOGO / "dev2"):
        prev |= {h["wnid"] for h in json.loads((work / "splits" / "heldout.json").read_text())}
    assert len(prev) == 200
    u1 = set()
    for k in range(1, 6):
        u1 |= set(json.loads((P4 / "banks" / "U1" / f"split{k}.json").read_text())["heldout"])
    splits = []
    for k in range(1, N_SPLITS + 1):
        held, info, stats = select_relaxed(wnids, clean, parents, SEED + k, N_HELD, frozenset(prev), 2)
        splits.append({"split": k, "seed": SEED + k, "heldout": held, "info": info, "stats": stats})
    held_any = set().union(*[set(s["heldout"]) for s in splits])
    assert not (held_any & prev)
    overlap = {f"{a['split']}-{b['split']}": len(set(a["heldout"]) & set(b["heldout"])) for i, a in enumerate(splits) for b in splits[i + 1:]}
    names, used_hashes, n_p4 = used_sets()
    sealed_paths, sealed_hashes, _ = load_sealed()
    G.update(names=names, used_hashes=used_hashes, sealed_paths=sealed_paths, sealed_hashes=sealed_hashes)
    base = N_SHOT + N_IDEVAL
    tasks = [(i, w, base + (N_OODEVAL if w in held_any else 0)) for i, w in enumerate(wnids)]
    with mp.get_context("fork").Pool(16) as pool:
        res = pool.map(pick_class, tasks, chunksize=4)
    count = collections.Counter(dg for _, _, picked, _, _ in res for _, _, dg in picked)
    cross = {dg for dg, n in count.items() if n > 1}
    rows, rejections = [], {}
    for idx, wnid, picked, rej, ncand in sorted(res, key=lambda x: x[0]):
        need = base + (N_OODEVAL if wnid in held_any else 0)
        keep = [x for x in picked if x[2] not in cross]
        if len(keep) < need:
            raise RuntimeError(f"{wnid}: {len(keep)} after cross-class duplicate removal")
        rej["cross_class_duplicate"] = len(picked) - len(keep)
        rejections[wnid] = rej
        for j, (name, real, dg) in enumerate(keep[:need]):
            if j < N_SHOT:
                role, draw = "shot", j // 16
            elif j < base:
                role, draw = "ideval", -1
            else:
                role, draw = "oodeval", -1
            rows.append({"sample_id": f"p5in1k/train/{wnid}/{name}", "path": real, "sha256": dg, "wnid": wnid, "idx_1k": idx,
                         "clean_name": clean[idx], "role": role, "slot": j, "draw": draw})
    pool_f = pd.DataFrame(rows)
    pos = np.full(len(pool_f), -1)
    for (idx, draw), sub in pool_f[pool_f.role == "shot"].groupby(["idx_1k", "draw"]):
        perm = np.random.default_rng([SEED, 2, int(idx), int(draw)]).permutation(len(sub))
        pos[sub.index.values[perm]] = np.arange(len(sub))
    pool_f["pos"] = pos
    pool_f["shot_role"] = np.where(pool_f.role != "shot", "", np.where(pool_f.pos < 12, "support", "calib"))
    assert pool_f.sha256.is_unique and not (set(pool_f.sha256) & used_hashes) and not (set(pool_f.sha256) & sealed_hashes)
    pool_f.to_parquet(out / "imagenet_pool.parquet", index=False)
    for s in splits:
        s["heldout_names"] = [clean[wnids.index(w)] for w in s["heldout"]]
        s["id_wnids"] = [w for w in wnids if w not in set(s["heldout"])]
        s["n_heldout_also_in_U1"] = len(set(s["heldout"]) & u1)
        del s["info"]
        dump(out / "U4" / f"split{s['split']}.json", s)
    pool_f[["sample_id", "path"]].to_parquet(out / "feature_table.parquet", index=False)
    info = {"utc": utc(), "seed": SEED, "lock_sha256": sha_file(out.parent / "selection_lock_p5.json"), "wordnet_version": wn_version,
            "n_pool": len(pool_f), "pool_roles": pool_f.role.value_counts().to_dict(), "n_heldout_union": len(held_any),
            "n_heldout_union_in_U1": len(held_any & u1), "split_overlap": overlap, "n_phase4_pool_excluded": n_p4,
            "n_used_hashes": len(used_hashes), "n_cross_class_duplicates": len(cross),
            "rejected": {k: int(sum(r.get(k, 0) for r in rejections.values())) for k in
                         ("sealed_path", "sealed_hash", "used_hash", "duplicate", "cross_class_duplicate")},
            "splits": [{"split": s["split"], "seed": s["seed"], "n_heldout": len(s["heldout"]),
                        "n_also_in_U1": s["n_heldout_also_in_U1"]} for s in splits],
            "sha256": {"imagenet_pool.parquet": sha_file(out / "imagenet_pool.parquet"),
                       "feature_table.parquet": sha_file(out / "feature_table.parquet")}}
    dump(out / "build_info.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
