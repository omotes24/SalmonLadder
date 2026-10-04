"""Phase 7 runner: one (experiment, stream, view) task = one frozen streaming run with every read-out of engine7.

Output: <P7>/results/<exp>/<view>/<stream>.npz  (log p of every read-out, admissions, rho; labels and classes for the
evaluation scripts only; MCM and, where it exists for exactly this stream, the TINS score).

Experiments (--exp):
  u1std   U1, 5 splits x 3 draws x 3 orders (standard streams, with the batch-only propagation)        [A, B, C]
  u2std   U2 (ImageNet-O), 5 draws x 3 orders                                                         [B]
  u4std   U4, 5 splits x 3 draws x 3 orders                                                           [A]
  u1alone U1, splits 1-5, draw 0, order 123, with the image-alone propagation                         [C3]
  u1r u4r the standard U1 / U4 streams with the delayed re-scoring read-outs (delays 1, 2, 5, 10, 20, end)  [A]
  u1w u4w the same streams with the within-batch memory read-out ('@0': members admitted from the same batch count,
          the image itself never does)                                                                [A, amendment 02]
  u2w e1w e2w e3w dsw   the tasks of u2std / e1 / e2 / e3 / ds with the '@0' read-out, so that the locked extension
          (selection_lock_p7.json) can be reported next to the frozen method everywhere                [B, D, E]
  ds      dataset banks: natural streams (3 orders) and the matched composition (3 seeds)             [D]
  e1 e2 e3  design space of stream conditions on U1 (draw 0)                                          [E]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data7
import engine7 as E7
import streams7 as S7
from p7common import RESULTS, SEED, V5, utc
from static import static_view

DATASETS = ("cub", "cifar100", "places365", "inr", "insk")


RETRO = (1, 2, 5, 10, 20)


def run_stream(P, ids, flag, bidx, view, device="cuda", lp_batch=False, lp_alone=False, logS=None, extra=None, retro=()):
    C = P.C
    K = min(V5["K"], C)
    clip_s, clip_c = P.feat("CLIP", ids), P.feat("CLIP", P.cal_ids)
    sims_s, sims_c = clip_s @ P.pos.T, clip_c @ P.pos.T
    cand_s = np.argsort(-sims_s, axis=1)[:, :K]
    cand_c = np.argsort(-sims_c, axis=1)[:, :K]
    mcm = torch.softmax(torch.as_tensor(sims_s, dtype=torch.float64), dim=1).max(1).values.numpy()
    sup = P.feat(view, P.sup_ids).reshape(C, 12, -1)
    cal = P.feat(view, P.cal_ids)
    sf = P.feat(view, ids)
    st = static_view(sup, cal, sf, cand_c, cand_s, V5["n0"], V5["m"])
    out, aux = E7.run_view7(sup, cal, sf, bidx, st, V5["thresholds"], k=V5["kg"], gamma=V5["gamma"], lam=V5["lam"], device=device,
                            lp_batch=lp_batch, lp_alone=lp_alone, retro=retro)
    arrays = {"sample_id": np.array(ids), "is_ood": np.asarray(flag, bool), "cls": P.cls(ids).astype(str), "bidx": np.asarray(bidx),
              "logM": np.log(mcm), "admit": aux["admit"], "rho": aux["rho"], "d": st["d"][len(cal):].astype(np.float32)}
    if logS is not None:
        arrays["logS"] = logS
    for k, v in out.items():
        arrays[f"s::{k}"] = v
    meta = {"problem": P.name, "view": view, "n": len(ids), "n_ood": int(np.sum(flag)), "C": C, "seconds": aux["seconds"], **(extra or {})}
    arrays["meta"] = np.array(json.dumps(meta))
    return arrays


def save(exp, view, name, arrays):
    d = RESULTS / exp / view
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, d / f"{name}.npz")


def tins_scores(P, seed, ids):
    z = np.load(P.tins(seed), allow_pickle=True)
    assert list(z["sample_id"]) == list(ids), "TINS stream differs"
    return np.log(z["S_final"].astype(np.float64)), z["batch_index"]


# ------------------------------------------------------------------------------------------------ task lists
def tasks_std(exp):
    if exp in ("u1std", "u1r", "u1w"):
        return [("u1", (s, d), sd) for s in range(1, 6) for d in range(3) for sd in (123, 124, 125)]
    if exp in ("u4std", "u4r", "u4w"):
        return [("u4", (s, d), sd) for s in range(1, 6) for d in range(3) for sd in (123, 124, 125)]
    if exp in ("u2std", "u2w"):
        return [("u2", (d,), sd) for d in range(5) for sd in (123, 124, 125)]
    if exp == "u1alone":
        return [("u1", (s, 0), 123) for s in range(1, 6)]
    raise ValueError(exp)


E1_CELLS = [(U, m) for U in (100, 30, 10) for m in (1, 2, 5, 10, 20, 50, 100, 200) if U * m >= 100]
E2_PATTERNS = ("random", "id_burst10", "ood_burst", "id_first", "ood_early", "emerging", "two_visits")
E2_COMP = {"native": (100, 50), "sparse": (100, 10)}


def tasks_e(exp):
    t = []
    if exp == "e1":
        for n_id in (18000, 4500):
            for (U, m) in E1_CELLS:
                for s in range(1, 6):
                    for seed in (1, 2):
                        t.append({"split": s, "U": U, "m": m, "n_id": n_id, "pattern": "random", "seed": seed, "batch": 256})
        for m in (1, 5, 20, 50, 200):
            for s in range(1, 6):
                t.append({"split": s, "U": 100, "m": m, "n_id": 72000, "pattern": "random", "seed": 1, "batch": 256})
    elif exp == "e2":
        for comp, (U, m) in E2_COMP.items():
            for pat in E2_PATTERNS:
                for s in range(1, 6):
                    t.append({"split": s, "U": U, "m": m, "n_id": 18000, "pattern": pat, "seed": 1, "batch": 256, "comp": comp})
    elif exp == "e3":
        for pat in ("random", "ood_burst"):
            for B in (1, 16, 64, 256):
                for s in range(1, 6):
                    t.append({"split": s, "U": 100, "m": 50, "n_id": 18000, "pattern": pat, "seed": 1, "batch": B})
    return t


def e_name(c):
    return f"s{c['split']}_U{c['U']}_m{c['m']}_n{c['n_id']}_{c['pattern']}_B{c['batch']}_seed{c['seed']}"


def e_stream(P, c):
    rng = np.random.default_rng([SEED, 71, c["split"], c["U"], c["m"], c["n_id"], c["seed"]])
    ide, ood = S7.compose(P, c["U"], c["m"], c["n_id"], rng)
    id_cls = P.cls(ide)
    ids, flag = S7.arrange(ide, ood, c["pattern"], np.random.default_rng([SEED, 72, c["split"], c["U"], c["m"], c["n_id"], c["seed"]]), id_cls)
    return ids, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--views", default="B14,L14")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--datasets", default=",".join(DATASETS) + ",cubssb")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="walk the task list backwards (a second queue meeting the first)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    views = a.views.split(",")
    exp = a.exp
    # '<exp>w': the same tasks with the within-batch memory read-out ('@0') added; every frozen read-out is unchanged
    base = exp[:-1] if exp in ("e1w", "e2w", "e3w", "dsw") else exp
    retro = RETRO if exp in ("u1r", "u4r") else ((0,) if exp.endswith("w") else ())
    jobs = []
    if exp in ("u1std", "u2std", "u4std", "u1alone", "u1r", "u4r", "u1w", "u4w", "u2w"):
        for (bank, key, seed) in tasks_std(exp):
            for v in views:
                jobs.append(("std", bank, key, seed, v))
    elif base == "ds":
        for dsn in a.datasets.split(","):
            if dsn == "cubssb":
                for lv in ("Easy", "Medium", "Hard"):
                    for seed in (123, 124, 125):
                        for v in views:
                            jobs.append(("cubssb", lv, None, seed, v))
            else:
                for s in range(1, 6):
                    for cond in ("nat", "mat"):
                        for seed in (123, 124, 125):
                            for v in views:
                                jobs.append(("ds", dsn, (s, cond), seed, v))
        for s in range(1, 6):                           # U1 under the matched composition (reference row)
            for seed in (123, 124, 125):
                for v in views:
                    jobs.append(("u1mat", "u1", (s, 0), seed, v))
    elif base in ("e1", "e2", "e3"):
        for c in tasks_e(base):
            for v in views:
                jobs.append(("e", c, None, None, v))
    else:
        raise ValueError(exp)
    if a.reverse:
        jobs = jobs[::-1]
    jobs = jobs[a.worker::a.nworkers]
    if a.limit:
        jobs = jobs[:a.limit]
    cache = {}

    def problem(kind, *key):
        k = (kind,) + key
        if k not in cache:
            cache.clear()
            cache[k] = getattr(data7, kind)(*key)
        return cache[k]

    for job in jobs:
        kind, view = job[0], job[4]
        t0 = time.time()
        if kind == "std":
            _, bank, key, seed, _ = job
            P = problem(bank, *key)
            name = f"{P.name}_seed{seed}"
            if (RESULTS / exp / view / f"{name}.npz").exists():
                continue
            ids, flag = S7.standard(P, seed)
            logS, bidx = tins_scores(P, seed, ids)
            arr = run_stream(P, ids, flag, bidx, view, lp_batch=(exp == "u1std"), lp_alone=(exp == "u1alone"), logS=logS, retro=retro)
        elif kind == "cubssb":
            _, lv, _, seed, _ = job
            P = problem("cub_ssb", lv)
            name = f"{P.name}_seed{seed}"
            if (RESULTS / exp / view / f"{name}.npz").exists():
                continue
            ids, flag = S7.standard(P, seed)
            logS, bidx = tins_scores(P, seed, ids)
            arr = run_stream(P, ids, flag, bidx, view, logS=logS, retro=retro)
        elif kind in ("ds", "u1mat"):
            _, dsn, (s, cond), seed, _ = job
            P = problem("ds", dsn, s) if kind == "ds" else problem("u1", s, 0)
            cond = "mat" if kind == "u1mat" else cond
            name = f"{P.name}_{cond}_seed{seed}"
            if (RESULTS / exp / view / f"{name}.npz").exists():
                continue
            extra = {"cond": cond}
            if cond == "nat":
                ids, flag = S7.standard(P, seed)
            else:
                ids, flag, info = S7.matched(P, seed)
                extra.update(info)
            arr = run_stream(P, ids, flag, np.arange(len(ids)) // 256, view, extra=extra, retro=retro)
        else:
            c = job[1]
            name = e_name(c)
            if (RESULTS / exp / view / f"{name}.npz").exists():
                continue
            P = problem("u1", c["split"], 0)
            ids, flag = e_stream(P, c)
            arr = run_stream(P, ids, flag, np.arange(len(ids)) // c["batch"], view, extra={"cell": c}, retro=retro)
        save(exp, view, name, arr)
        n_rows = int(len(arr["is_ood"]))
        del arr
        torch.cuda.empty_cache()
        print(json.dumps({"exp": exp, "view": view, "task": name, "n": n_rows, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    main()
