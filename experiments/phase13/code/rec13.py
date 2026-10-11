"""Phase 13 (c2): the recurrence x mixing-rate grid of Phase 12 (same test sets: same random choice of classes and images) with
the product read-outs added: p+ p- x kNN rank, p+ p- x prototype rank, p+ x prototype rank, and the plain prototype rank alone.
Output: P13/results/rec/<ds>/c<c>_r<r>_rep<i>.csv and .meta.json."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p13common as P
import core12 as C
import engine5 as E5
from engine import _log
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L")]
GRID = {"ninco": [(c, r) for c in (5, 10, 20, 40, 64) for r in (1, 2, 5, 10, 20, 30)],
        "ssb_hard": [(c, r) for c in (10, 30, 100, 300, 980) for r in (1, 2, 5, 10, 20, 50)]}
REPS = 3
READOUTS = {"T": ["lp", "ln|q0.1"], "Tq0.2": ["lp", "ln|q0.2"], "Tlp": ["lp"], "knn": ["knn"], "static": ["static"], "proto": ["proto"],
            "Tq0.2xknn": ["lp", "ln|q0.2", "knn"], "Tq0.2xproto": ["lp", "ln|q0.2", "proto"], "Tlpxproto": ["lp", "proto"], "Tq0.2xstatic": ["lp", "ln|q0.2", "static"]}


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def classes_of(part="openood"):
    """{sample id: class folder} of the NINCO and SSB-hard images, from the image paths (features.pt order = D['ids'] order after the shots)."""
    paths = P.image_paths(part)[16000:]
    return paths


def run(worker, nworkers, device="cuda"):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    cand = D["cands"][P.V5["K"]]
    paths = classes_of()
    ids = list(D["ids"])
    assert len(paths) == len(ids)
    cls = {}
    for s, pth in zip(ids, paths):
        parts = Path(pth).parts
        cls[s] = (parts[-3], parts[-2])
    # the ID images and the OOD images of the two data sets, from the seed-123 stream files (TINS scores come with them)
    sets = {}
    for ds, seed, path in P.stream_tasks("openood"):
        if ds not in ("ninco", "ssb_hard") or seed != 123:
            continue
        z = np.load(path, allow_pickle=True)
        sid, flag, S = z["sample_id"], z["is_ood"].astype(bool), np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        sets[ds] = {"id": sid[~flag], "ood": sid[flag], "logS": dict(zip(sid, S))}
    id_ids = sets["ninco"]["id"]
    assert set(id_ids) == set(sets["ssb_hard"]["id"]) and len(id_ids) == 45000
    from static import support_dall
    from vins import r5
    shots = {v: (l2n(D["shots"][v][:12000]), l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    tasks = [(ds, c, r, rep) for ds in GRID for (c, r) in GRID[ds] for rep in range(REPS)][worker::nworkers]
    for ds, c, r, rep in tasks:
        out_dir = P.RESULTS13 / "rec" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"c{c}_r{r}_rep{rep}"
        if (out_dir / f"{name}.csv").exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])
        by_class = defaultdict(list)
        for s in sets[ds]["ood"]:
            by_class[cls[s][1]].append(s)
        eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
        if len(eligible) < c:
            continue
        chosen = rng.choice(eligible, size=c, replace=False)
        ood_ids = []
        groups = []
        for k in chosen:
            pick = rng.choice(np.array(by_class[k]), size=r, replace=False)
            groups.append((k, len(ood_ids), len(ood_ids) + r))
            ood_ids.extend(pick.tolist())
        test_ids = np.concatenate([id_ids, np.array(ood_ids)])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        logS = np.array([sets[ds]["logS"][x] for x in test_ids])
        feats = {v: (shots[v][0], shots[v][1], l2n(D["ev"][v][rows - 4000])) for v in VIEWS}
        terms, meta = {}, {"ds": ds, "c": c, "r": r, "rep": rep, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)),
                           "mixing_rate": float(len(ood_ids) / len(test_ids)), "views": {}}
        for v, (sup, cal, sf) in feats.items():
            g = C.Graph(sup, cal, sf, device)
            out, m = C.read_out(g, flag)
            cuts = [g.cut(g.nfix + len(id_ids) + np.arange(a, b)) for (_, a, b) in groups[:50]]     # at most 50 classes measured
            m["cut"] = {k: float(np.mean([cu[k] for cu in cuts])) for k in cuts[0]} if cuts else None
            # support mass received by the unknown classes vs ID (sqrt-degree weighted sums are in the proposition; report the plain means)
            u = out["u_pos"]
            m["u_pos_mean"] = {"id": float(u[g.nfix:g.nfix + len(id_ids)].mean()), "ood": float(u[g.nfix + len(id_ids):].mean()),
                               "cal": float(u[g.ns:g.nfix].mean())}
            g.free()
            terms[(v, "lp")] = out["lp"]
            for q in P.QS:
                terms[(v, f"ln|q{q:g}")] = out[f"ln|q{q:g}"]
            terms[(v, "knn")] = C.knn_rank(sup, cal, sf, dev=device)
            terms[(v, "proto")] = P.proto_rank(sup, cal, sf, dev=device)
            if v in VIEWS:
                st = E5.static5(r5.proto_view, support_dall, sup.reshape(1000, 12, -1), cal, sf, cand[:4000], cand[rows], P.V5["n0"], P.V5["m"])
                terms[(v, "static")] = _log(np.asarray(st["p"])[len(cal):])
            meta["views"][v] = m
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in READOUTS.items():
                if any((v, k) not in terms for v in vs for k in ks):
                    continue
                x = sum(terms[(v, k)] for v in vs for k in ks)
                for base, add in (("none", 0.0), ("TINS", logS)):
                    au, fp = measures(x + add, flag)
                    rows_out.append({"ds": ds, "c": c, "r": r, "rep": rep, "n_ood": len(ood_ids), "mixing_rate": meta["mixing_rate"],
                                     "views": "+".join(vs), "readout": ro, "base": base, "AUROC": au, "FPR95": fp})
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        d3 = meta["views"]["D3L"]
        print(json.dumps({"ds": ds, "c": c, "r": r, "rep": rep, "n_ood": len(ood_ids), "seeds_q0.1": d3["q0.1"], "cut": d3["cut"],
                          "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
