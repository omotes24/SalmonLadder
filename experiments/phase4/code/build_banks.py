"""Build the unused evaluation banks of Phase 4 (no image is scored here; no test image is opened).

U1  ImageNet-1K train, five new class splits: 100 held-out classes each (every held-out class keeps a WordNet
    sibling in ID; at most two held-out classes per parent; seeds 20260928+k), disjoint from the dev1/dev2 held-out
    classes; different new splits may share classes (amendments 01-03). Images never used by REPRISE:
    dev1/dev2 samples, the TINS 16-shot list and Codex's replaced duplicates, the five R5 draws, sealed images.
    Per class: 80 shots (5 draws x 12 support + 4 calibration), 20 ID-evaluation, 60 long-stream images,
    and 50 OOD-evaluation images for classes held out in a U1 split.
U2  ImageNet-O (official, 2,000 images) as OOD; ID = 10 ID-evaluation images of every ImageNet-1K class.
U3  WILD (5 ID / 5 OOD wildlife species, duplicate-excluded split of the user's other project, unused by REPRISE):
    80 shots per ID class from train/id, ID = test/id, OOD = test/ood (400 per class in the main stream).
"""
import collections
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import BANKS, GONOGO, IMAGENET, IMAGENET_O, WILD, R5, dump, sha_file, utc
from vins import config as C
from vins.sealing import load_sealed
from vins.splits import list_class_files, load_imagenet_classes, wordnet_info

SEED = 20260928
N_SPLITS, N_HELD = 5, 100
N_SHOT, N_IDEVAL, N_LONG, N_OODEVAL, N_SPARE = 80, 20, 60, 50, 12
G = {}


def select_relaxed(wnids, clean, parents, seed, n, exclude, max_per_parent=2):
    """dev1/dev2 rule (a held-out class keeps an ID sibling) with at most `max_per_parent` held-out classes per
    WordNet parent (dev1/dev2 used one; one-per-parent is infeasible once the 200 dev1/dev2 held-out classes and
    the earlier new splits are excluded: amendment_01)."""
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
    stats = {"rule": f"ID sibling kept, <= {max_per_parent} held-out per parent", "n_candidates": len(cand),
             "selection_order": held, "excluded": sorted(exclude)}
    return held, info, stats


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
    return names, hashes, len(r5)


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
    out = BANKS
    out.mkdir(parents=True, exist_ok=True)
    if (out / "imagenet_pool.parquet").exists():
        raise SystemExit("banks already built; refusing to overwrite")
    wnids, raw, clean = load_imagenet_classes()
    parents, lemmas, wn_version = wordnet_info(wnids)
    prev = set()
    for work in (GONOGO, GONOGO / "dev2"):
        prev |= {h["wnid"] for h in json.loads((work / "splits" / "heldout.json").read_text())}
    assert len(prev) == 200
    splits = []
    for k in range(1, N_SPLITS + 1):
        held, info, stats = select_relaxed(wnids, clean, parents, SEED + k, N_HELD, frozenset(prev), max_per_parent=2)
        splits.append({"split": k, "seed": SEED + k, "heldout": held, "info": info, "stats": stats})
    held_any = set().union(*[set(s["heldout"]) for s in splits])
    assert not (held_any & prev)
    overlap = {f"{a['split']}-{b['split']}": len(set(a["heldout"]) & set(b["heldout"])) for i, a in enumerate(splits) for b in splits[i + 1:]}
    names, used_hashes, n_r5 = used_sets()
    sealed_paths, sealed_hashes, _ = load_sealed()
    G.update(names=names, used_hashes=used_hashes, sealed_paths=sealed_paths, sealed_hashes=sealed_hashes)
    base = N_SHOT + N_IDEVAL + N_LONG
    tasks = [(i, w, base + (N_OODEVAL if w in held_any else 0)) for i, w in enumerate(wnids)]
    with mp.get_context("fork").Pool(20) as pool:
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
        keep = keep[:need]
        for j, (name, real, dg) in enumerate(keep):
            if j < N_SHOT:
                draw, within = divmod(j, 16)
                role, extra = "shot", {"draw": draw}
            elif j < N_SHOT + N_IDEVAL:
                role, extra = "ideval", {"draw": -1}
            elif j < base:
                role, extra = "long", {"draw": -1}
            else:
                role, extra = "oodeval", {"draw": -1}
            rows.append({"sample_id": f"p4in1k/train/{wnid}/{name}", "path": real, "sha256": dg, "wnid": wnid,
                         "idx_1k": idx, "clean_name": clean[idx], "role": role, "slot": j, **extra})
    pool_f = pd.DataFrame(rows)
    # support / calibration assignment inside each draw: seeded permutation (12 + 4), as R5 but a new seed
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
        s["heldout_detail"] = [{"wnid": w, "clean_name": clean[wnids.index(w)], "parents": s["info"][w]["parents"],
                                "id_siblings": s["info"][w]["id_siblings"], "lemmas": lemmas[w]} for w in s["heldout"]]
        del s["info"]
        dump(out / "U1" / f"split{s['split']}.json", s)
    # U2: ImageNet-O
    o_files = sorted(p for p in IMAGENET_O.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file())
    o_rows = []
    for p in o_files:
        real = os.path.realpath(p)
        o_rows.append({"sample_id": f"imagenet_o/{p.relative_to(IMAGENET_O)}", "path": real, "sha256": sha_file(real),
                       "wnid": p.parent.name})
    o = pd.DataFrame(o_rows)
    o = o[~o.sha256.isin(sealed_hashes | used_hashes | set(pool_f.sha256))].drop_duplicates("sha256").reset_index(drop=True)
    o.to_parquet(out / "U2_imagenet_o.parquet", index=False)
    # U3: WILD
    man = pd.read_csv(WILD / "file_manifest.csv")
    w_rows = []
    id_species = sorted(man[(man.group == "id")].species.unique())
    ood_species = sorted(man[(man.group == "ood")].species.unique())
    for sp in id_species:
        tr = man[(man.split == "train") & (man.group == "id") & (man.species == sp)].sort_values("relative_path")
        perm = np.random.default_rng([SEED, 3, id_species.index(sp)]).permutation(len(tr))[:N_SHOT]
        for j, r in enumerate(tr.iloc[perm].itertuples()):
            draw = j // 16
            w_rows.append({"sample_id": f"wild/{r.relative_path}", "path": os.path.realpath(r.path), "species": sp,
                           "group": "id", "role": "shot", "draw": draw, "slot": j})
    for r in man[(man.split == "test")].itertuples():
        w_rows.append({"sample_id": f"wild/{r.relative_path}", "path": os.path.realpath(r.path), "species": r.species,
                       "group": r.group, "role": "ideval" if r.group == "id" else "oodeval", "draw": -1, "slot": -1})
    w = pd.DataFrame(w_rows)
    pos = np.full(len(w), -1)
    for (sp, draw), sub in w[w.role == "shot"].groupby(["species", "draw"]):
        perm = np.random.default_rng([SEED, 4, id_species.index(sp), int(draw)]).permutation(len(sub))
        pos[sub.index.values[perm]] = np.arange(len(sub))
    w["pos"] = pos
    w["shot_role"] = np.where(w.role != "shot", "", np.where(w.pos < 12, "support", "calib"))
    main_ood = np.zeros(len(w), bool)
    for sp in ood_species:
        ix = w.index[(w.role == "oodeval") & (w.species == sp)].values
        main_ood[np.random.default_rng([SEED, 5, ood_species.index(sp)]).permutation(ix)[:400]] = True
    w["main_stream"] = np.where(w.role == "ideval", True, main_ood)
    w["sha256"] = [sha_file(p) for p in w.path]
    dup = w.sha256.duplicated(keep=False)
    w["duplicate"] = dup
    w.to_parquet(out / "U3_wild.parquet", index=False)
    table = pd.concat([pool_f[["sample_id", "path"]].assign(source="imagenet"),
                       o[["sample_id", "path"]].assign(source="imagenet_o"),
                       w[["sample_id", "path"]].assign(source="wild")]).drop_duplicates("sample_id").reset_index(drop=True)
    table.to_parquet(out / "feature_table.parquet", index=False)
    info = {"utc": utc(), "seed": SEED, "wordnet_version": wn_version, "n_pool": len(pool_f),
            "pool_roles": pool_f.role.value_counts().to_dict(), "n_imagenet_o": len(o), "n_wild": len(w),
            "wild_id_species": id_species, "wild_ood_species": ood_species, "wild_exact_duplicates": int(dup.sum()),
            "n_feature_rows": len(table), "n_r5_excluded": n_r5, "n_used_hashes": len(used_hashes),
            "n_cross_class_duplicates": len(cross), "n_heldout_union": len(held_any), "split_overlap": overlap,
            "rejected": {k: int(sum(r.get(k, 0) for r in rejections.values())) for k in
                         ("sealed_path", "sealed_hash", "used_hash", "duplicate", "cross_class_duplicate")},
            "splits": [{"split": s["split"], "seed": s["seed"], "n_heldout": len(s["heldout"])} for s in splits],
            "sha256": {"imagenet_pool.parquet": sha_file(out / "imagenet_pool.parquet"),
                       "U2_imagenet_o.parquet": sha_file(out / "U2_imagenet_o.parquet"),
                       "U3_wild.parquet": sha_file(out / "U3_wild.parquet"),
                       "feature_table.parquet": sha_file(out / "feature_table.parquet")}}
    dump(out / "build_info.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
