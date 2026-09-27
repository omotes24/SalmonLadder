"""R5 (2): entrances compared at matched ID admission rates (dev1 tunes, dev2 evaluates; no test data).

Entrance families (thresholds shared by the two views, as in v4; every set is chosen among all stream samples):
  E1  static one-stage            M  = {p_all <= e1}
  E2  A1 -> M                     A1 = {p_all <= e1},  M = {p_A1 <= e2}
  E3  A1 -> A2 -> M (v4 form)     A2 = {p_A1 <= e2},   M = {p_A2 <= e3}
--phase tune   (dev1): final-stage p-values of every configuration of the earlier stages, 15 conditions x near/far.
--phase select        : per family and target ID admission rate (1%, 5%, 10%): for every configuration the final
                        threshold that makes the pooled dev1 ID admission rate closest to the target; among
                        configurations the one with the highest dev1 near-OOD admission rate is frozen (frozen.json).
                        The same is done for E3 with neighbour ranks m = 1 and 5 (all stages and the memory score).
--phase eval   (dev2; dev1 for reference): ID admission rate (admitted ID / arrived ID), OOD admission rate,
                        memory OOD purity (OOD share of M at the end), detection with the memory only
                        (S x p_t, both views) and with LP (S x p_t x p_LP, LP from r5_eval), alone and x TINS.
--phase inject (dev2, diagnostic): n in {1, 2, 5} arrivals of one ID class forced into M; later arrivals of that
                        class: rate of p_t <= alpha with and without the injection, for m in {1, 2, 5}.
Labels are used for tuning on dev1 and for measurement only; the frozen thresholds never see dev2 labels.
Outputs: <R5>/entrance/...
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.dview import loo_stats  # noqa: E402

BASE = Path.home() / "vins_gonogo_20260925"
OUT = r5.R5 / "entrance"
TARGETS = (0.01, 0.05, 0.10)
E2_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
E3_GRID = [(a, b) for a in (0.2, 0.3, 0.4, 0.5, 0.6) for b in (0.1, 0.2, 0.3, 0.4, 0.5)]
E3M_GRID = [(a, b) for a in (0.3, 0.4, 0.5) for b in (0.2, 0.3, 0.4)]
DRAWS = ("0", "1", "2", "3", "4")
G = {}


def load_r5_eval():
    spec = importlib.util.spec_from_file_location("r5_eval", ROOT / "scripts" / "r5_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_work(dev):
    os.environ["VINS_WORK"] = str(BASE if dev == "dev1" else BASE / "dev2")
    import importlib

    importlib.reload(C)


def scale(v, m):
    if m == 2:
        return v["stats"]["med_all"], v["stats"]["mad_all"]
    st = loo_stats(v["support_arr"], m, C.N0, C.MAD_SCALE)
    return st["med_all"], st["mad_all"]


def stream_inputs(dev, draw, stream, seed):
    z = np.load(r5.R5 / dev / "tins" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
    e = np.load(r5.R5 / dev / "eval" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
    assert (z["sample_id"] == e["sample_id"]).all()
    return z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64), z["batch_index"], e


def view_inputs(v, q):
    return v["q"][q], v["d"][q], v["p_all"][q], v["cal_feats"], v["d_cal"]


def tune_task(task):
    dev, draw, stream, seed = task
    sid, is_ood, S, bidx, _ = stream_inputs(dev, draw, stream, seed)
    q = np.array([G["row"][s] for s in sid])
    arrays = {}
    for name, v in G["views"].items():
        sf, d, p_all, cf, dc = view_inputs(v, q)
        arrays[f"{name}|E1"] = p_all
        for m in (2, 1, 5):
            med, mad = scale(v, m)
            e1s = E2_GRID if m == 2 else sorted({a for a, _ in E3M_GRID})
            grid = E3_GRID if m == 2 else E3M_GRID
            pa1 = {}
            for e1 in e1s:
                pa1[e1], _ = r5.memory_p(sf, d, bidx, p_all <= e1, cf, dc, med, mad, m)
                if m == 2:
                    arrays[f"{name}|E2|{e1}"] = pa1[e1]
            for e1, e2 in grid:
                pa2, _ = r5.memory_p(sf, d, bidx, pa1[e1] <= e2, cf, dc, med, mad, m)
                arrays[f"{name}|E3m{m}|{e1}|{e2}"] = pa2
    out = OUT / "tune" / f"{dev}_draw{draw}_{stream}_seed{seed}.npz"
    np.savez_compressed(out, is_ood=is_ood, **{k: a.astype(np.float32) for k, a in arrays.items()})
    return task


def select():
    files = sorted((OUT / "tune").glob("dev1_draw*_*.npz"))
    assert len(files) == 30, len(files)
    data = [(f, np.load(f)) for f in files]
    keys = [k for k in data[0][1].files if k != "is_ood"]
    fams = {}
    for k in keys:
        name, fam, *cfg = k.split("|")
        fams.setdefault(fam, set()).add(tuple(cfg))
    frozen, table = {}, {}
    for fam, cfgs in fams.items():
        for tau in TARGETS:
            rows = []
            for cfg in sorted(cfgs):
                key = "|".join([fam, *cfg])
                p_id = np.concatenate([z[f"{n}|{key}"][~z["is_ood"]] for _, z in data for n in ("B14", "L14")])
                cand = np.unique(p_id)
                rates = np.searchsorted(np.sort(p_id), cand, side="right") / len(p_id)
                j = int(np.argmin(np.abs(rates - tau) + 1e-12 * cand))
                thr, rate = float(cand[j]), float(rates[j])
                near = np.concatenate([z[f"{n}|{key}"][z["is_ood"]] for f, z in data if "_near_" in f.name
                                       for n in ("B14", "L14")])
                far = np.concatenate([z[f"{n}|{key}"][z["is_ood"]] for f, z in data if "_far_" in f.name
                                      for n in ("B14", "L14")])
                rows.append({"cfg": [float(x) for x in cfg], "final_threshold": thr, "id_rate": rate,
                             "near_ood_rate": float((near <= thr).mean()), "far_ood_rate": float((far <= thr).mean())})
            best = max(rows, key=lambda r: r["near_ood_rate"])
            frozen[f"{fam}@{tau:g}"] = {"family": fam, "target": tau, "thresholds": best["cfg"] + [best["final_threshold"]],
                                        "m": int(fam[3:]) if fam.startswith("E3m") else 2,
                                        "dev1_id_rate": best["id_rate"], "dev1_near_ood_rate": best["near_ood_rate"],
                                        "dev1_far_ood_rate": best["far_ood_rate"]}
            table[f"{fam}@{tau:g}"] = rows
    frozen["v4"] = {"family": "E3m2", "target": None, "thresholds": list(r5.V4_ENTRANCE), "m": 2}
    blob = json.dumps(frozen, indent=1) + "\n"
    (OUT / "frozen.json").write_text(blob)
    (OUT / "tune_table.json").write_text(json.dumps(table, indent=1) + "\n")
    print(blob)
    print("frozen.json sha256", hashlib.sha256(blob.encode()).hexdigest())


def measures(id_s, ood_s):
    from vins.metrics import measures as upstream

    m = upstream(id_s, ood_s)
    return {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}


def eval_task(task):
    dev, draw, stream, seed = task
    sid, is_ood, S, bidx, e = stream_inputs(dev, draw, stream, seed)
    q = np.array([G["row"][s] for s in sid])
    res = {}
    for key, cfg in G["frozen"].items():
        th, m = cfg["thresholds"], cfg["m"]
        pts, stats = [], {}
        for name, v in G["views"].items():
            sf, d, p_all, cf, dc = view_inputs(v, q)
            med, mad = scale(v, m)
            masks = r5.entrance(sf, d, p_all, bidx, cf, dc, med, mad, tuple(th), m)
            M = masks[-1]
            pt, _ = r5.memory_p(sf, d, bidx, M, cf, dc, med, mad, m)
            pts.append(pt)
            stats[name] = {"id_rate": float(M[~is_ood].mean()), "ood_rate": float(M[is_ood].mean()),
                           "purity": float(is_ood[M].mean()) if M.any() else float("nan"), "size": int(M.sum()),
                           "stage_id_rates": [float(x[~is_ood].mean()) for x in masks]}
        pt = pts[0] * pts[1]
        lp = e["plp_B14"].astype(np.float64) * e["plp_L14"].astype(np.float64)
        sc = {"s_mem": S * pt, "s_full": S * pt * lp, "v_mem": pt, "v_full": pt * lp}
        res[key] = {"stats": stats, "metrics": {k: measures(s[~is_ood], s[is_ood]) for k, s in sc.items()}}
    out = OUT / "eval" / f"{dev}_draw{draw}_{stream}_seed{seed}.json"
    out.write_text(json.dumps(res) + "\n")
    return task


def inject_task(task):
    dev, draw, stream, seed = task
    sid, is_ood, S, bidx, _ = stream_inputs(dev, draw, stream, seed)
    q = np.array([G["row"][s] for s in sid])
    rng = np.random.default_rng([7, int(draw)])
    cls_of = G["cls"]
    id_rows = np.flatnonzero(~is_ood)
    classes = rng.choice(np.unique(cls_of[q[id_rows]]), 40, replace=False)
    res = {}
    for name, v in G["views"].items():
        sf, d, p_all, cf, dc = view_inputs(v, q)
        for m in (1, 2, 5):
            med, mad = scale(v, m)
            M = r5.entrance(sf, d, p_all, bidx, cf, dc, med, mad, r5.V4_ENTRANCE, m)[-1]
            base_all, _ = r5.memory_p(sf, d, bidx, M, cf, dc, med, mad, m)
            for c in classes:
                arr = id_rows[cls_of[q[id_rows]] == c]              # arrivals of class c in stream order
                for n in (1, 2, 5):
                    inj = arr[:n]
                    later = arr[bidx[arr] > bidx[inj[-1]]]
                    if len(later) < 3:
                        continue
                    M2 = M.copy()
                    M2[inj] = True
                    base = base_all[later]
                    hit = r5.memory_p_rows(sf, d, bidx, M2, cf, dc, med, mad, m, later)
                    key = f"{name}|m{m}|n{n}"
                    rec = res.setdefault(key, {"n_classes": 0, "n_eval": 0, "base_le05": 0.0, "inj_le05": 0.0,
                                               "base_le10": 0.0, "inj_le10": 0.0, "base_mean_p": 0.0, "inj_mean_p": 0.0,
                                               "already_in_memory": 0})
                    rec["n_classes"] += 1
                    rec["n_eval"] += len(later)
                    rec["already_in_memory"] += int(M[inj].sum())
                    for tag, p in (("base", base), ("inj", hit)):
                        rec[f"{tag}_le05"] += float((p <= 0.05).sum())
                        rec[f"{tag}_le10"] += float((p <= 0.10).sum())
                        rec[f"{tag}_mean_p"] += float(p.sum())
    out = OUT / "inject" / f"{dev}_draw{draw}_{stream}_seed{seed}.json"
    out.write_text(json.dumps(res) + "\n")
    return task


def prepare(dev, draw):
    set_work(dev)
    ev_mod = load_r5_eval()
    views, row = ev_mod.build(draw, ev_mod.load_ev(), baselines=False)
    G.update(views=views, row=row)
    if "cls" not in G or G.get("cls_dev") != dev:
        import pandas as pd

        samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
        stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
        cls = samples.set_index("sample_id").loc[stream_ids].class_idx_id.values.astype(int)
        n_cal = len(next(iter(views.values()))["d_cal"])
        G["cls"] = np.concatenate([np.full(n_cal, -1), cls])
        G["cls_dev"] = dev


def run_phase(phase, devs, workers, draws=DRAWS):
    fn = {"tune": tune_task, "eval": eval_task, "inject": inject_task}[phase]
    (OUT / phase).mkdir(parents=True, exist_ok=True)
    if phase in ("eval",):
        G["frozen"] = json.loads((OUT / "frozen.json").read_text())
        from vins.tins_dev import import_tins

        import_tins()
    for dev in devs:
        for draw in draws:
            prepare(dev, draw)
            streams = ("near",) if phase == "inject" else ("near", "far")
            seeds = (123,) if phase == "inject" else C.ORDER_SEEDS
            tasks = [(dev, draw, s, sd) for s in streams for sd in seeds]
            with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
                for t in pool.imap_unordered(fn, tasks):
                    print(json.dumps({"done": t}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["tune", "select", "eval", "inject"], required=True)
    parser.add_argument("--devs", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--draws", nargs="+", default=list(DRAWS))
    opts = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if opts.phase == "select":
        select()
    else:
        devs = opts.devs or {"tune": ["dev1"], "eval": ["dev2", "dev1"], "inject": ["dev2"]}[opts.phase]
        run_phase(opts.phase, devs, opts.workers, opts.draws)
    print(json.dumps({"phase": opts.phase, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
