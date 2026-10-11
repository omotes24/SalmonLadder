"""Phase 17 smoke test (run before the registration is placed; prints checks only, no metric of sigma_ref or of the descriptive
analyses): (1) dev1 draw 0 near: the paper's sigma recomputed in this pipeline equals the Phase 16 run of the same task (rows of
sigma and p_fin), the internal recomputation of sigma equals score_all's, sigma_ref runs (seed counts only); (2) one grid task
(NINCO c = 5, r = 10, rep 0): the three bounds run and are never violated (sigma <= B, h <= pi_Q, pi_Q <= phibar); (3) the
covariate-shift features are complete. Output: P17/results/smoke17.json."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p17common as P
import run17 as R

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"utc_start": P.utc()}
    # (1) dev1 draw 0 near
    D = dev5.load("dev1")
    groups = defaultdict(list)
    for (draw, stream, seed) in sorted(D["tins"].keys()):
        groups[(draw, stream)].append(seed)
    draw, stream = next(k for k in sorted(groups) if str(k[0]) == "0" and str(k[1]) == "near")
    seeds = groups[(draw, stream)]
    sid0, flag0, _, _ = D["tins"][(draw, stream, seeds[0])]
    order = np.argsort(np.asarray(sid0))
    ids, flag = np.asarray(sid0)[order], np.asarray(flag0, bool)[order]
    rows = np.array([D["stream_row"][s] for s in ids])
    raw = {v: dev5.raw_view(D, draw, v, rows) for v in R.VIEWS}
    sup_cal = {v: (P.l2n(raw[v][0].reshape(-1, raw[v][0].shape[-1])), P.l2n(raw[v][1])) for v in R.VIEWS}
    graphs, terms, feats = R.build(sup_cal, lambda v: P.l2n(raw[v][2]), "cuda")
    n_classes = len(sup_cal[R.VIEWS[0]][0]) // 12
    t0 = time.time()
    rows_out, meta, arrays = R.score_task(graphs, terms, feats, flag, {}, np.repeat(np.arange(n_classes), 12), n_classes, {"dev": "dev1"}, "cuda", decomp=True)
    for g in graphs.values():
        g.free()
    d = pd.DataFrame(rows_out)
    p16 = pd.read_csv(P.RESULTS16 / "h16_dev" / "dev1" / "dev1_draw0_near.csv")
    chk = {}
    for cfg in (R.X.SIG, R.X.PF, "xfit|lp|pooled"):
        a = d[(d.cfg == cfg) & (d.base == "none")].iloc[0]
        b = p16[(p16.cfg == cfg) & (p16.base == "none")].iloc[0]
        chk[cfg] = {"dFPR95": float(abs(a.FPR95 - b.FPR95)), "dAUROC": float(abs(a.AUROC - b.AUROC))}
    out["dev1_draw0_near"] = {"phase16_equal": chk, "all_equal": bool(all(v["dFPR95"] < 1e-4 and v["dAUROC"] < 1e-4 for v in chk.values())),   # percent units
                              "sigma_recomputed_maxdiff": meta["sigma_recomputed_maxdiff"], "ref_seeds": meta["ref"]["views"],
                              "finite_ref": bool(np.isfinite(arrays["ref_s"]).all() and np.isfinite(arrays["ref_pfin"]).all()),
                              "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out["dev1_draw0_near"], indent=1), flush=True)
    # (2) one grid task with the bounds
    Dt = P.load_p3("openood")
    P.add_views(Dt, "openood", ["D3L"])
    paths = P.image_paths("openood")[16000:]
    cls = {s: Path(pth).parts[-2] for s, pth in zip(list(Dt["ids"]), paths)}
    sets = {}
    for ds, seed, path in P.stream_tasks("openood"):
        if ds != "ninco" or seed != 123:
            continue
        z = np.load(path, allow_pickle=True)
        sid, fl = z["sample_id"], z["is_ood"].astype(bool)
        sets[ds] = {"id": sid[~fl], "ood": sid[fl]}
    c, r, rep = 5, 10, 0
    rng = np.random.default_rng([12, c, r, rep, 0])
    by_class = defaultdict(list)
    for s in sets["ninco"]["ood"]:
        by_class[cls[s]].append(s)
    eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
    chosen = rng.choice(eligible, size=c, replace=False)
    ood_ids, ood_cls = [], []
    for j, k in enumerate(chosen):
        pick = rng.choice(np.array(by_class[k]), size=r, replace=False).tolist()
        ood_ids.extend(pick)
        ood_cls.extend([j] * len(pick))
    id_ids = sets["ninco"]["id"]
    test_ids = np.concatenate([id_ids, np.array(ood_ids)])
    fl = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
    cls_of_test = np.r_[np.full(len(id_ids), -1), np.array(ood_cls)]
    rws = np.array([Dt["row"][x] for x in test_ids])
    shots = {v: (P.l2n(Dt["shots"][v][:12000]), P.l2n(Dt["shots"][v][12000:16000])) for v in R.VIEWS}
    graphs, terms, feats = R.build(shots, lambda v: P.l2n(Dt["ev"][v][rws - 4000]), "cuda")
    t0 = time.time()
    _, meta, arrays = R.score_task(graphs, terms, feats, fl, {}, np.repeat(np.arange(1000), 12), 1000, {"ds": "ninco"}, "cuda", bounds=cls_of_test)
    for g in graphs.values():
        g.free()
    out["grid_ninco_c5_r10_rep0"] = {"violations": meta["bounds"], "n_unknown": int(len(arrays["c2_B"])), "n_a1": int(len(arrays["a1_pi"])),
                                     "finite": bool(np.isfinite(arrays["c2_B"]).all() and np.isfinite(arrays["c2r_B"]).all()), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out["grid_ninco_c5_r10_rep0"], indent=1), flush=True)
    # (3) covariate-shift features
    cs = {}
    for name in P.CSID_ROOTS:
        t = pd.read_parquet(P.TABLES17 / f"{name}.parquet")
        sh = {v: list(np.load(P.FEATS17 / name / f"{v}.npy", mmap_mode="r").shape) for v in R.VIEWS}
        cs[name] = {"rows": int(len(t)), "features": sh}
    out["csid_features"] = cs
    out["ok"] = bool(out["dev1_draw0_near"]["all_equal"] and out["dev1_draw0_near"]["finite_ref"] and out["grid_ninco_c5_r10_rep0"]["finite"]
                     and all(v == 0 for v in out["grid_ninco_c5_r10_rep0"]["violations"].values())
                     and all(cs[n]["features"][v][0] == cs[n]["rows"] for n in cs for v in R.VIEWS))
    out["utc_end"] = P.utc()
    P.write_json(P.RESULTS17 / "smoke17.json", out)
    print(json.dumps({"ok": out["ok"]}), flush=True)


if __name__ == "__main__":
    main()
