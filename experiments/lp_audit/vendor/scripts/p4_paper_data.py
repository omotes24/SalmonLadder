"""Post-hoc analyses for the paper figures (reads the frozen Phase 3 outputs; nothing is selected or tuned).

1. base detectors: MCM / NegLabel / TINS x the REPRISE (v5) visual part, and the visual part alone
   (OpenOOD 5 orders, Four-OOD 3 orders), from the saved per-sample p-values.
2. recurrence on test (SSB-hard, NINCO): OOD detection at the ID-95% threshold by the number of earlier
   same-class images in the stream, and by stream position (first occurrences only / per bin).
3. score space: a fixed subsample of (log TINS score, log visual evidence) for one order.
4. conformal calibration on test: Pr(p <= a | ID) for the static, memory and LP p-values.
Output: <R5>/paper/paper_data.json
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
OUT = r5.R5 / "paper"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
S5, S3 = [123, 124, 125, 126, 127], [123, 124, 125]
ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
KBINS = [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]


def prod(z, key):
    return z[f"{key}_B14"].astype(np.float64) * z[f"{key}_L14"].astype(np.float64)


def main():
    from vins.tins_dev import import_tins

    import_tins()
    from vins.metrics import measures

    OUT.mkdir(exist_ok=True)
    res = {"note": "post-hoc; frozen v5 outputs of Phase 3; no selection"}
    # 1. base detectors -------------------------------------------------------------------------------------
    base = {}
    for part, dss, seeds in (("openood", OO, S5), ("fourood", FOUR, S3)):
        bl = np.load(C.WORK / "analysis" / "baselines" / f"{part}.npz")
        pos = {s: i for i, s in enumerate(bl["sample_id"].tolist())}
        for ds in dss:
            for sd in seeds:
                z = np.load(P3 / part / f"{ds}_seed{sd}.npz", allow_pickle=True)
                is_ood = z["is_ood"].astype(bool)
                k = np.array([pos[x] for x in z["sample_id"].tolist()])
                vis = prod(z, "v5_pt") * prod(z, "v5_plp")
                sc = {"mcm": bl["mcm"][k].astype(np.float64), "neglabel": bl["neglabel"][k].astype(np.float64),
                      "tins": z["S"].astype(np.float64)}
                for b in ("mcm", "neglabel", "tins"):
                    sc[f"{b}_x_v5"] = sc[b] * vis
                sc["v5_vis"] = vis
                ref = json.loads((P3 / part / f"{ds}_seed{sd}.json").read_text())["metrics"]
                m = {}
                for name, s in sc.items():
                    mm = measures(s[~is_ood], s[is_ood])
                    m[name] = {"AUROC": 100 * mm["AUROC"], "FPR95": 100 * mm["FPR95"]}
                m["check_tins_x_v5_minus_json_v5_AUROC"] = m["tins_x_v5"]["AUROC"] - ref["v5"]["AUROC"]
                base[f"{part}|{ds}|{sd}"] = m
    res["base_detectors"] = base
    # 2. recurrence on test -----------------------------------------------------------------------------------
    spec = importlib.util.spec_from_file_location("p3_streams", ROOT / "scripts" / "p3_streams.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    tab = ps.test_table()
    cls_of = dict(zip(tab.sample_id, tab.cls))
    rec = {}
    for ds in ("ssb_hard", "ninco"):
        for sd in S5:
            z = np.load(P3 / "openood" / f"{ds}_seed{sd}.npz", allow_pickle=True)
            ids = z["sample_id"].tolist()
            is_ood = z["is_ood"].astype(bool)
            n = len(ids)
            S = z["S"].astype(np.float64)
            sc = {"tins": S, "static": S * prod(z, "v5_p"), "memory": S * prod(z, "v5_pt"), "lp": S * prod(z, "v5_plp"),
                  "full": S * prod(z, "v5_pt") * prod(z, "v5_plp")}
            seen, kk = {}, np.full(n, -1)
            for i, (s, o) in enumerate(zip(ids, is_ood)):
                if o:
                    c = cls_of[s]
                    kk[i] = seen.get(c, 0)
                    seen[c] = kk[i] + 1
            posq = np.arange(n) / n
            out = {"n_ood": int(is_ood.sum()), "n_classes": len(seen)}
            for name, s in sc.items():
                tau = np.quantile(s[~is_ood], 0.05)
                det = s < tau
                row = {}
                for lo, hi in KBINS:
                    sel = is_ood & (kk >= lo) & (kk <= hi)
                    if sel.sum() == 0:
                        continue
                    mm = measures(s[~is_ood], s[sel])
                    row[f"{lo}-{hi}"] = {"n": int(sel.sum()), "det": float(det[sel].mean()), "AUROC": 100 * mm["AUROC"],
                                         "mean_pos": float(posq[sel].mean()),
                                         "det_by_posq": [float(det[sel & (posq >= a) & (posq < a + 0.25)].mean())
                                                         if (sel & (posq >= a) & (posq < a + 0.25)).sum() else None
                                                         for a in (0, 0.25, 0.5, 0.75)],
                                         "n_by_posq": [int((sel & (posq >= a) & (posq < a + 0.25)).sum()) for a in (0, 0.25, 0.5, 0.75)]}
                out[name] = row
            rec[f"{ds}|{sd}"] = out
    res["recurrence_test"] = rec
    # 3. score space (order 123) ------------------------------------------------------------------------------
    rng = np.random.default_rng(20260927)
    space = {}
    for ds in ("ssb_hard", "ninco", "openimageo"):
        z = np.load(P3 / "openood" / f"{ds}_seed123.npz", allow_pickle=True)
        is_ood = z["is_ood"].astype(bool)
        S = z["S"].astype(np.float64)
        vis = prod(z, "v5_pt") * prod(z, "v5_plp")
        full = S * vis
        th = {k: float(np.quantile(v[~is_ood], 0.05)) for k, v in (("tins", S), ("vis", vis), ("full", full))}
        pick = {}
        for lab, mask in (("id", ~is_ood), ("ood", is_ood)):
            idx = np.flatnonzero(mask)
            idx = rng.choice(idx, min(3000, len(idx)), replace=False)
            pick[lab] = {"logS": np.round(np.log10(S[idx]), 4).tolist(),
                         "logV": np.round(np.log10(np.maximum(vis[idx], 1e-300)), 4).tolist()}
        space[ds] = {"thresholds": th, **pick,
                     "frac_ood_caught_by_full_missed_by_tins": float(((full < th["full"]) & (S >= th["tins"]))[is_ood].mean()),
                     "frac_ood_caught_by_tins_missed_by_full": float(((full >= th["full"]) & (S < th["tins"]))[is_ood].mean())}
    res["score_space"] = space
    # 4. calibration on test -----------------------------------------------------------------------------------
    cal = {}
    for part, dss, seeds in (("openood", OO, S5), ("fourood", FOUR, S3)):
        acc = {}
        for ds in dss:
            for sd in seeds:
                z = np.load(P3 / part / f"{ds}_seed{sd}.npz", allow_pickle=True)
                idm = ~z["is_ood"].astype(bool)
                for key in ("v5_p", "v5_pt", "v5_plp"):
                    for v in ("B14", "L14"):
                        x = z[f"{key}_{v}"][idm].astype(np.float64)
                        acc.setdefault(f"{key}_{v}", []).append([float((x <= a).mean()) for a in ALPHAS])
        cal[part] = {"alphas": ALPHAS, **{k: np.mean(v, axis=0).tolist() for k, v in acc.items()}}
    res["calibration_test"] = cal
    (OUT / "paper_data.json").write_text(json.dumps(res) + "\n")
    print(json.dumps({"written": str(OUT / "paper_data.json"),
                      "max_abs_check": max(abs(v["check_tins_x_v5_minus_json_v5_AUROC"]) for v in base.values())}))


if __name__ == "__main__":
    main()
