"""R5 Phase 2 round 2 (pre-registered: r5/prereg_phase2_round2.json). dev only; test data are never read.

Coordinate-wise search on dev1 over n0 -> |K| -> m -> gamma -> lambda -> kg, starting from v4, on the 15 matched
conditions; score = S_final x prod_views p_t x prod_views p_LP. Views are vins.r5.proto_view (equal to the frozen
custom_view for n0 = 12); candidate sets K(x) = CLIP zero-shot top-|K| over the dev ID classes (TINS positive text
features); m = neighbour rank of the memory score, with the entrance thresholds 0.40/0.30/0.10 (m = 2) or the frozen
dev1 tuning 0.30/0.20/0.1019 (m = 1). LP = vins.r5.lp_run(k = kg, alpha = lambda, gamma, 15 warm sweeps).
Per factor the level with the lowest mean dev1 near FPR95 among levels whose mean dev1 far FPR95 difference to v4 is
<= +0.5 is kept. The final configuration is then evaluated on dev2 together with v4 and the round-1 v5 (m = 1).
p-values are cached per (task, setting) under <R5>/<dev>/round2/cache.
Output: <R5>/round2/search.json, decision.json
"""
import argparse
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import r5  # noqa: E402

BASE = Path.home() / "vins_gonogo_20260925"
DRAWS = ["0", "1", "2", "3", "4"]
SEEDS = [123, 124, 125]
STREAMS = ["near", "far"]
V4 = {"n0": 12, "K": 5, "m": 2, "gamma": 3.0, "lam": 0.9, "kg": 10}
ORDER = [("n0", [12, 24, 48]), ("K", [5, 10, 20]), ("m", [2, 1]), ("gamma", [0.0, 0.5, 1.0, 2.0, 3.0]),
         ("lam", [0.7, 0.75, 0.8, 0.85, 0.9]), ("kg", [5, 7, 10, 15])]
THRESH = {2: (0.40, 0.30, 0.10), 1: (0.30, 0.20, 0.10191613435745239)}
G = {}


def set_dev(dev):
    os.environ["VINS_WORK"] = str(BASE if dev == "dev1" else BASE / "dev2")
    import importlib

    from vins import config

    importlib.reload(config)
    return config


def load_dev(dev):
    """Features, CLIP zero-shot similarities, TINS scores and row maps of one dev split (all draws)."""
    C = set_dev(dev)
    import importlib.util

    spec = importlib.util.spec_from_file_location("r5_eval", ROOT / "scripts" / "r5_eval.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    from vins.features import load_features

    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    n_id = len(id_classes)
    stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
    pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
    clip_dev, cb = load_features(C.FEATURES_DIR / "clip.pt")
    cpos = {s: i for i, s in enumerate(cb["sample_id"])}
    shot_clip, sb = load_features(r5.R5 / "features" / "shots.clip.pt")
    spos = {s: i for i, s in enumerate(sb["sample_id"])}
    st_sims = (clip_dev[[cpos[s] for s in stream_ids]].float() @ pos_text.T).numpy()
    k5 = np.array(dview.loc[stream_ids].K_id.tolist())
    agree = float(np.mean([set(a) == set(b) for a, b in zip(np.argsort(-st_sims, axis=1)[:, :5], k5)]))
    feats = {}
    for name, dev_file, shot_file in E.VIEWS:
        f, b = load_features(C.FEATURES_DIR / dev_file)
        p = {s: i for i, s in enumerate(b["sample_id"])}
        sh, shb = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
        shp = {s: i for i, s in enumerate(shb["sample_id"])}
        feats[name] = {"stream": f[[p[s] for s in stream_ids]].numpy().astype(np.float32), "shots": sh.numpy(),
                       "shot_pos": shp}
    draws = {}
    for d in DRAWS:
        sup_ids, cal_ids, _ = E.shot_table(d, id_classes, samples)
        cal_sims = (shot_clip[[spos[s] for s in cal_ids]].float() @ pos_text.T).numpy()
        draws[d] = {"sup": sup_ids, "cal": cal_ids, "cal_sims": cal_sims}
    tins = {}
    for d in DRAWS:
        for s in STREAMS:
            for sd in SEEDS:
                z = np.load(r5.R5 / dev / "tins" / f"draw{d}" / f"{s}_seed{sd}.npz", allow_pickle=True)
                tins[(d, s, sd)] = (z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64),
                                    z["batch_index"])
    return {"dev": dev, "n_id": n_id, "stream_row": {s: i for i, s in enumerate(stream_ids)}, "st_sims": st_sims,
            "k5": k5, "k5_agree": agree, "feats": feats, "draws": draws, "tins": tins}


def view_inputs(D, d, name, n0, K, m):
    F, dr = D["feats"][name], D["draws"][d]
    sup = F["shots"][[F["shot_pos"][s] for s in dr["sup"]]].astype(np.float32).reshape(D["n_id"], 12, -1)
    cal = F["shots"][[F["shot_pos"][s] for s in dr["cal"]]].astype(np.float32)
    q = np.concatenate([cal, F["stream"]])
    if K == 5:
        cand = np.concatenate([np.argsort(-dr["cal_sims"], axis=1)[:, :5], D["k5"]])   # stream: dview K_id (as v4)
    else:
        cand = np.concatenate([np.argsort(-dr["cal_sims"], axis=1)[:, :K], np.argsort(-D["st_sims"], axis=1)[:, :K]])
    is_cal = np.zeros(len(q), dtype=bool)
    is_cal[:len(cal)] = True
    v = r5.proto_view(sup, q, cand, is_cal, n0=n0, m=m)
    return v, sup, q, len(cal)


def cache_path(dev, kind, task, setting):
    d, s, sd = task
    tag = "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in sorted(setting.items()))
    return r5.R5 / dev / "round2" / "cache" / f"{kind}_d{d}_{s}_{sd}_{tag}.npz"


def job(args):
    kind, task, setting = args
    D = G["D"]
    out = cache_path(D["dev"], kind, task, setting)
    if out.exists():
        return str(out)
    d, s, sd = task
    sid, is_ood, S, bidx = D["tins"][task]
    rows = np.array([D["stream_row"][x] for x in sid])
    res = {}
    for name in ("B14", "L14"):
        if kind == "mem":
            v, sup, q, nc = view_inputs(D, d, name, setting["n0"], setting["K"], setting["m"])
            qi = nc + rows
            sf, dd, p_all = q[qi], v["d"][qi], v["p_all"][qi]
            med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
            M = r5.entrance(sf, dd, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, THRESH[setting["m"]], setting["m"])[-1]
            res[name], _ = r5.memory_p(sf, dd, bidx, M, v["cal_feats"], v["d_cal"], med, mad, setting["m"])
        else:
            F, dr = D["feats"][name], D["draws"][d]
            sup = F["shots"][[F["shot_pos"][x] for x in dr["sup"]]].astype(np.float32).reshape(D["n_id"], 12, -1)
            sup /= np.linalg.norm(sup, axis=2, keepdims=True)
            cal = F["shots"][[F["shot_pos"][x] for x in dr["cal"]]].astype(np.float32)
            res[name] = r5.lp_run(sup, cal, F["stream"][rows], bidx, k=setting["kg"], alpha=setting["lam"],
                                  gamma=setting["gamma"], iters=15)["p"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: v.astype(np.float64) for k, v in res.items()})
    os.replace(tmp, out)
    return str(out)


def mem_setting(c):
    return {"n0": c["n0"], "K": c["K"], "m": c["m"]}


def lp_setting(c):
    return {"gamma": float(c["gamma"]), "lam": float(c["lam"]), "kg": c["kg"]}


def ensure(configs, workers):
    D = G["D"]
    tasks = [(d, s, sd) for d in DRAWS for s in STREAMS for sd in SEEDS]
    jobs = set()
    for c in configs:
        for t in tasks:
            for kind, st in (("mem", mem_setting(c)), ("lp", lp_setting(c))):
                if not cache_path(D["dev"], kind, t, st).exists():
                    jobs.add((kind, t, tuple(sorted(st.items()))))
    jobs = [(k, t, dict(st)) for k, t, st in sorted(jobs)]
    if jobs:
        with mp.get_context("fork").Pool(workers) as pool:
            for _ in pool.imap_unordered(job, jobs):
                pass
    return len(jobs)


def metrics(c):
    """{(draw, stream, seed): {FPR95, AUROC}} of configuration c."""
    from vins.metrics import measures as upstream

    D = G["D"]
    out = {}
    for d in DRAWS:
        for s in STREAMS:
            for sd in SEEDS:
                t = (d, s, sd)
                _, is_ood, S, _ = D["tins"][t]
                pm = np.load(cache_path(D["dev"], "mem", t, mem_setting(c)))
                pl = np.load(cache_path(D["dev"], "lp", t, lp_setting(c)))
                sc = S * pm["B14"] * pm["L14"] * pl["B14"] * pl["L14"]
                m = upstream(sc[~is_ood], sc[is_ood])
                out[t] = {"FPR95": 100 * m["FPR95"], "AUROC": 100 * m["AUROC"]}
    return out


def draw_means(met, stream, key):
    return [float(np.mean([met[(d, stream, sd)][key] for sd in SEEDS])) for d in DRAWS]


def diff(a, b, stream, key):
    return r5.t_interval([x - y for x, y in zip(draw_means(a, stream, key), draw_means(b, stream, key))])


def summary(met):
    return {f"{s}_{k}": float(np.mean(draw_means(met, s, k))) for s in STREAMS for k in ("FPR95", "AUROC")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    opts = parser.parse_args()
    start = time.time()
    from vins.tins_dev import import_tins

    import_tins()
    out_dir = r5.R5 / "round2"
    out_dir.mkdir(parents=True, exist_ok=True)
    G["D"] = load_dev("dev1")
    log = {"k5_agreement_dev1": G["D"]["k5_agree"], "rounds": []}
    state = dict(V4)
    ensure([state], opts.workers)
    v4m = metrics(V4)
    ref = json.loads((r5.R5 / "summary" / "stageA.json").read_text())["dev1"]["table"]["s_full"]
    log["v4_reproduction_dev1"] = {"round2": summary(v4m), "stageA_near_FPR95": ref["near_FPR95"]["mean"],
                                   "stageA_near_AUROC": ref["near_AUROC"]["mean"]}
    for factor, levels in ORDER:
        configs = [dict(state, **{factor: lv}) for lv in levels]
        n_jobs = ensure(configs, opts.workers)
        rows = []
        for c in configs:
            met = metrics(c)
            dfar = diff(met, v4m, "far", "FPR95")["mean"]
            rows.append({"level": c[factor], **summary(met), "far_FPR95_minus_v4": dfar,
                         "near_FPR95_minus_v4": diff(met, v4m, "near", "FPR95")})
        ok = [r for r in rows if r["far_FPR95_minus_v4"] <= 0.5]
        cur = next(r for r in rows if r["level"] == state[factor])
        best = min(ok, key=lambda r: (r["near_FPR95"], r["level"] != state[factor])) if ok else cur
        state[factor] = best["level"]
        log["rounds"].append({"factor": factor, "rows": rows, "chosen": best["level"], "jobs": n_jobs,
                              "seconds": round(time.time() - start, 1)})
        print(json.dumps({"factor": factor, "chosen": best["level"], "near_FPR95": best["near_FPR95"]}), flush=True)
        (out_dir / "search.json").write_text(json.dumps(log, indent=1, default=float) + "\n")
    final = dict(state)
    fm1 = metrics(final)
    dec = {"final": final, "dev1": {"final": summary(fm1), "v4": summary(v4m),
                                    "near_FPR95_diff": diff(fm1, v4m, "near", "FPR95"),
                                    "far_FPR95_diff": diff(fm1, v4m, "far", "FPR95"),
                                    "near_AUROC_diff": diff(fm1, v4m, "near", "AUROC")}}
    dev1_ok = dec["dev1"]["near_FPR95_diff"]["hi"] < 0
    # dev2: final, v4 and the round-1 v5 (m = 1)
    G["D"] = load_dev("dev2")
    r1 = dict(V4, m=1)
    ensure([V4, final, r1], opts.workers)
    m_v4, m_f, m_r1 = metrics(V4), metrics(final), metrics(r1)
    dec["dev2"] = {"final": summary(m_f), "v4": summary(m_v4), "round1_v5": summary(m_r1),
                   "near_FPR95_diff": diff(m_f, m_v4, "near", "FPR95"), "far_FPR95_diff": diff(m_f, m_v4, "far", "FPR95"),
                   "near_AUROC_diff": diff(m_f, m_v4, "near", "AUROC"),
                   "final_minus_round1_near_FPR95": diff(m_f, m_r1, "near", "FPR95"),
                   "final_minus_round1_far_FPR95": diff(m_f, m_r1, "far", "FPR95")}
    ok2 = dec["dev2"]["near_FPR95_diff"]["hi"] < 0 and dec["dev2"]["far_FPR95_diff"]["mean"] <= 0.5
    if dev1_ok and ok2:
        dec["verdict"] = "confirmed: v5 = round-2 configuration"
        dec["v5"] = final
    else:
        dec["verdict"] = "not confirmed: v5 = round-1 decision (v4 + m=1 entrance)"
        dec["v5"] = r1
    dec["seconds"] = round(time.time() - start, 1)
    (out_dir / "decision.json").write_text(json.dumps(dec, indent=1, default=float) + "\n")
    print(json.dumps({"verdict": dec["verdict"], "final": final}))


if __name__ == "__main__":
    main()
