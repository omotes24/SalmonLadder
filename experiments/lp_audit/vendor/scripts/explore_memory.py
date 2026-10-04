"""Exploratory (dev1 only): test-time visual OOD memory (vins.memory) fused with TINS as S_final * p_t.

Reads the frozen view and the TINS logs of the main run (3 order seeds); TINS is not re-run.
Memory admission is decided from frozen quantities only (no feedback from the memory itself):
  pall<e> : p_all <= e %, d_all = min over ALL ID classes of z_c (far from every ID class)
  all     : every earlier stream sample (unlabelled memory; --stream-all)
Score kinds (OOD direction), recalibrated per batch on the calibration set: diff, boost, lrat (vins.memory).
Outputs: <WORK>/iter/<tag>/memory.json and report.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.candidates import class_r_all  # noqa: E402
from vins.dview import loo_stats  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.memory import memory_distances, online_pvalues, pvalues_sorted  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

KINDS = ("diff", "boost", "lrat")
M_MEM = (1, 2)


def frozen_views(m):
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / "dino.pt")
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    n_id = 1000 - C.N_HELDOUT
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(n_id, C.N_SUPPORT, -1).astype(np.float64)
    stats = loo_stats(support, m, C.N0, C.MAD_SCALE)
    q = feats[row_of.loc[dview.sample_id].values].astype(np.float64)
    z_all = (class_r_all(q, support, m) - stats["med_t"][None, :]) / stats["mad_t"][None, :]
    k_idx = np.array(dview.K_id.tolist())
    d_zs = np.take_along_axis(z_all, k_idx, axis=1).min(axis=1)
    assert np.allclose(d_zs, dview[f"d_dino_m{m}"].values, atol=1e-8), "frozen view mismatch"
    d_all = z_all.min(axis=1)
    is_cal = (dview.split == "calib").values
    cal_zs, cal_all = np.sort(d_zs[is_cal]), np.sort(d_all[is_cal])
    info = pd.DataFrame({"d_zs": d_zs, "d_all": d_all, "p_zs": pvalues_sorted(cal_zs, d_zs),
                         "p_all": pvalues_sorted(cal_all, d_all)}, index=dview.sample_id.values)
    return {"feats": feats, "row_of": row_of, "stats": stats, "info": info, "cal_feats": q[is_cal].astype(np.float32),
            "d_cal": d_zs[is_cal], "cal_zs": cal_zs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.5, 1, 2])
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument("--mmem", type=int, nargs="+", default=list(M_MEM))
    parser.add_argument("--stream-all", action="store_true")
    opts = parser.parse_args()
    start = time.time()
    import_tins()
    out = C.WORK / "iter" / opts.tag
    out.mkdir(parents=True, exist_ok=True)
    fv = frozen_views(opts.m)
    med_all, mad_all = fv["stats"]["med_all"], fv["stats"]["mad_all"]
    per_seed, mem_stats = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
            is_id = (f.group == "ID").values
            s = f.S_final.values.astype(np.float64)
            inf = fv["info"].loc[f.sample_id.values]
            d = inf.d_zs.values
            bidx = f.batch_index.values
            sf = fv["feats"][fv["row_of"].loc[f.sample_id.values].values]
            p0 = pvalues_sorted(fv["cal_zs"], d)
            res = {"tins": measures(s[is_id], s[~is_id]), "clavis": measures((s * p0)[is_id], (s * p0)[~is_id]),
                   "d_only": measures(-d[is_id], -d[~is_id])}
            pa = inf.p_all.values
            admits = {f"pall{e:g}": pa <= e / 100 for e in opts.eps}
            if opts.stream_all:
                admits["all"] = np.ones(len(pa), dtype=bool)
            for name, adm in admits.items():
                r_s, r_c, batches = memory_distances(sf, bidx, adm, fv["cal_feats"], m_max=max(opts.mmem))
                mem_stats[f"{stream}|{seed}|{name}"] = {"n": int(adm.sum()), "n_ID": int((adm & is_id).sum()),
                                                         "n_OOD": int((adm & ~is_id).sum()),
                                                         "frac_OOD_admitted": float(adm[~is_id].mean())}
                for kind in opts.kinds:
                    for mm in opts.mmem:
                        p, g = online_pvalues(kind, mm, d, fv["d_cal"], r_s, r_c, bidx, batches, med_all, mad_all)
                        fused = s * p
                        res[f"{name}|{kind}|m{mm}"] = measures(fused[is_id], fused[~is_id])
                        res[f"{name}|{kind}|m{mm}|vis"] = measures(-g[is_id], -g[~is_id])
            entry[stream] = res
            print(json.dumps({"seed": seed, "stream": stream, "clavis": res["clavis"]["AUROC"],
                              "t": round(time.time() - start, 1)}), flush=True)
        per_seed.append(entry)
    agg = aggregate(per_seed)
    (out / "memory.json").write_text(json.dumps({"aggregate": agg, "memory": mem_stats, "m": opts.m,
                                                 "seconds": round(time.time() - start, 1)}, indent=1) + "\n")

    def a(stream, key, metric="AUROC"):
        return 100 * agg[stream][key][metric]["mean"]

    keys = [k for k in agg["near"] if not k.endswith("|vis") and k not in ("tins", "d_only")]
    keys.sort(key=lambda k: -a("near", k))
    lines = [f"# 視覚OODメモリの探索（dev1、事後解析、m={opts.m}）", "",
             "AUROC / FPR95（%、3 シード平均）。融合 = S_final × p_t。vis = 視覚スコア g 単体。", "",
             f"- TINS: near {a('near', 'tins'):.2f} / far {a('far', 'tins'):.2f}",
             f"- d 単体: near {a('near', 'd_only'):.2f} / far {a('far', 'd_only'):.2f}", "",
             "| 設定 | near AUROC | near FPR95 | far AUROC | far FPR95 | vis near | vis far |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        vis = f"{a('near', k + '|vis'):.2f} | {a('far', k + '|vis'):.2f}" if k + "|vis" in agg["near"] else "- | -"
        lines.append(f"| {k} | {a('near', k):.2f} | {a('near', k, 'FPR95'):.2f} | {a('far', k):.2f} | "
                     f"{a('far', k, 'FPR95'):.2f} | {vis} |")
    lines += ["", "## メモリの構成（最終時点、シード平均）", "", "| ストリーム | 採用規則 | 件数 | うち ID | OOD の採用率 |",
              "|---|---|---|---|---|"]
    for stream in ("near", "far"):
        for name in sorted({k.split("|")[2] for k in mem_stats}):
            rows = [mem_stats[f"{stream}|{s}|{name}"] for s in C.ORDER_SEEDS]
            lines.append(f"| {stream} | {name} | {np.mean([r['n'] for r in rows]):.0f} | "
                         f"{np.mean([r['n_ID'] for r in rows]):.0f} | {100 * np.mean([r['frac_OOD_admitted'] for r in rows]):.1f}% |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
