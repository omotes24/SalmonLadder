"""Phase 11 final: Salmon Ladder with the post-stream re-reading (遡上後の再採点) on every public-benchmark stream.
Per view v in {B14, L14, D3B, D3L, L14xD3L (= L2-normalised concatenation of L14 and D3L)} one graph over
[support (12,000); calibration (4,000); stream] with k_g = 10 (W = max(A, A^T), S = D^-1/2 W D^-1/2), lambda = 0.9:
  u+  converged mass from the support images         -> p+ = calibrated rank (p_low) among the calibration images
  seeds = Storey-BH(q) on p+ of the stream images     (q = 0.1 main; 0.05 and 0.2 for the ablation)
  u-  converged mass from the seeds on the same graph -> p- = calibrated rank (p_high)
  NN  closeness to the nearest seed (engine's g; ablation only) -> rank (p_high)
  k_g = 20 variant of (u+, seeds, u-) for the ablation.
Read-outs (sum over the views of a view set of the log calibrated ranks, + log base):
  T       log p+ + log p-      (q 0.1, k_g 10)   [main]
  Tlp     log p+
  Tq0.05 / Tq0.2               (seeds at q 0.05 / 0.2)
  Tnn     T + log NN
  Tk20    T on the k_g = 20 graph
  SL      streaming read-out of the paper (p_M x p_LP) from the stored streaming results (B14/L14/D3B/D3L only)
  static  static p of the stored streaming results
Bases: none, TINS, TANL (log scores of the same stream).
Per stream: P11/results/final/<part>/<stream>.npz (s::<view>::<key> test arrays, is_ood, sample_id, bases) and
            P11/results/final/<part>/<stream>.csv (AUROC / FPR95 of every views|readout|base, upstream measures)
            P11/results/final/<part>/<stream>.meta.json (number of seeds and their realised ID fraction per view and q)
--aggregate: P11/results/final/p11_summary.json (format of analysis10: near/far/fourood/per_dataset per cfg, paired t-CIs
             vs the reference = main configuration) and p11_table.txt."""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p11common as P
import engine5 as E5
from engine import _log
from sweep11 import l2n, storey_bh
from sweep11d import ViewGraph
from an11 import measures, tci

VIEWS = ["B14", "L14", "D3B", "D3L", "L14xD3L"]
Q_MAIN, QS, KG, K2 = 0.1, (0.05, 0.1, 0.2), 10, 20
VIEWSETS = [("L14xD3L", "D3L"), ("L14", "D3L"), ("D3L",), ("L14",), ("L14xD3L",), ("D3B", "D3L"), ("L14", "D3B", "D3L"),
            ("B14", "L14", "D3L"), ("B14", "L14", "D3B", "D3L"), ("B14", "L14")]
READOUTS = {
    "T": ["lp10", f"neg10|q{Q_MAIN:g}"],
    "Tlp": ["lp10"],
    "Tq0.05": ["lp10", "neg10|q0.05"],
    "Tq0.2": ["lp10", "neg10|q0.2"],
    "Tnn": ["lp10", f"neg10|q{Q_MAIN:g}", f"nn|q{Q_MAIN:g}"],
    "Tk20": ["lp20", f"neg20|q{Q_MAIN:g}"],
    "Tlp20": ["lp20"],
    "SL": ["M0", f"lp0|{P.REFT}"],
    "static": ["static"],
}
STREAM_ONLY = {"SL", "static"}            # read-outs that exist only for the four single backbone views
BASES = ("none", "TINS", "TANL")
MAIN = ("L14xD3L", "D3L"), "T", "TINS"
REF = "L14xD3L+D3L|T|TINS"


def cfg_name(views, ro, base):
    return "+".join(views) + "|" + ro + "|" + base


@torch.no_grad()
def view_terms(g, flag):
    """All per-view terms of the stream images (log calibrated ranks) and the seed statistics."""
    out, meta = {}, {}
    lp_t, _ = g.lp()                                   # k_g = 10, lambda 0.9, support mass
    out["lp10"] = lp_t
    p = np.exp(lp_t)
    for q in QS:
        idx = np.flatnonzero(storey_bh(p, q))
        meta[f"q{q:g}"] = {"seeds": int(len(idx)), "id_frac": float(1.0 - flag[idx].mean()) if len(idx) else float("nan")}
        if len(idx) == 0:
            out[f"neg10|q{q:g}"] = np.zeros_like(lp_t)
            if q == Q_MAIN:
                out[f"nn|q{q:g}"] = np.zeros_like(lp_t)
            continue
        out[f"neg10|q{q:g}"] = g.ranks_high(g.mass(KG, g.nfix + idx))[0]
        if q == Q_MAIN:
            out[f"nn|q{q:g}"] = g.nn_seed(idx)[0]
    # k_g = 20 variant
    lp20_t, _ = g.ranks_low(g.mass(K2, np.arange(g.ns)))
    out["lp20"] = lp20_t
    idx = np.flatnonzero(storey_bh(np.exp(lp20_t), Q_MAIN))
    meta[f"k20_q{Q_MAIN:g}"] = {"seeds": int(len(idx)), "id_frac": float(1.0 - flag[idx].mean()) if len(idx) else float("nan")}
    out[f"neg20|q{Q_MAIN:g}"] = g.ranks_high(g.mass(K2, g.nfix + idx))[0] if len(idx) else np.zeros_like(lp_t)
    return out, meta


def stream_terms(part, name, ids):
    """Streaming read-out terms (M0, lp0, static) of the four backbone views from the Phase 11 copies of the streaming results."""
    f = P.RESULTS11 / part / f"{name}.npz"
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=True)
    assert np.array_equal(z["sample_id"], ids)
    out = {}
    for v in ["B14", "L14", "D3B", "D3L"]:
        for key in ("M0", f"lp0|{P.REFT}", "static"):
            k = f"s::{v}::{key}"
            if k in z.files:
                out[(v, key)] = z[k].astype(np.float64)
    return out


def run_part(part, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / "final" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.csv").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    P.add_views(D, part, ["D3B", "D3L"])
    cand = D["cands"][P.V5["K"]]
    from static import support_dall
    from vins import r5
    feats, stat = {}, {}
    for v in ["B14", "L14", "D3B", "D3L"]:
        sup = D["shots"][v][:12000].reshape(1000, 12, -1)
        cal, ev = D["shots"][v][12000:16000], D["ev"][v]
        stat[v] = E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], P.V5["n0"], P.V5["m"])
        feats[v] = (D["shots"][v][:12000], cal, ev)
    feats["L14xD3L"] = tuple(l2n(np.concatenate([l2n(feats["L14"][i]), l2n(feats["D3L"][i])], 1)) for i in range(3))
    stat["L14xD3L"] = stat["D3L"]        # closeness statistics (used by the NN ablation term only)
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        logS = np.log(np.maximum(run["S_final"].astype(np.float64), 1e-300))
        zv = np.load(P.RESULTS / "vlm" / part / f"{name}.npz", allow_pickle=True)
        assert np.array_equal(zv["sample_id"], ids)
        logT = np.log(np.maximum(zv["TANL"].astype(np.float64), 1e-300))
        bases = {"none": 0.0, "TINS": logS, "TANL": logT}
        rows = np.array([D["row"][x] for x in ids])
        terms, meta = {}, {"part": part, "stream": name, "n": int(len(ids)), "n_ood": int(flag.sum()), "views": {}}
        for v in VIEWS:
            sup, cal, ev = feats[v]
            g = ViewGraph(sup, cal, ev[rows - 4000], stat[v], rows, dev)
            out, m = view_terms(g, flag)
            g.free()
            for key, x in out.items():
                terms[(v, key)] = np.asarray(x, np.float64)
            meta["views"][v] = m
        terms.update(stream_terms(part, name, ids))
        # scores and measures
        csv_rows, arrays = [], {"sample_id": np.array(ids), "is_ood": flag, "logTINS": logS, "logTANL": logT}
        for (v, key), x in terms.items():
            arrays[f"s::{v}::{key}"] = np.asarray(x, np.float32)
        for vs in VIEWSETS:
            for ro, keys in READOUTS.items():
                if ro in STREAM_ONLY and any((v, keys[0]) not in terms for v in vs):
                    continue
                if any((v, k) not in terms for v in vs for k in keys):
                    continue
                x = sum(terms[(v, k)] for v in vs for k in keys)
                for b, add in bases.items():
                    au, fp = measures(x + add, flag)
                    csv_rows.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": b, "AUROC": au, "FPR95": fp})
                    if (vs, ro, b) == MAIN:
                        arrays["main"] = np.asarray(x + add, np.float32)
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        pd.DataFrame(csv_rows).to_csv(out_dir / f"{name}.csv", index=False)
        main = [r for r in csv_rows if (tuple(r["views"].split("+")), r["readout"], r["base"]) == MAIN][0]
        print(json.dumps({"part": part, "stream": name, "rows": len(csv_rows), "main": [round(main["AUROC"], 2), round(main["FPR95"], 2)],
                          "seeds": {v: meta["views"][v][f"q{Q_MAIN:g}"] for v in MAIN[0]}, "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


def aggregate():
    base = P.RESULTS11 / "final"
    files = [f for part in ("openood", "fourood") for f in sorted(glob.glob(str(base / part / "*.csv")))]
    df = pd.concat([pd.read_csv(f) for f in files])
    df["ds"] = df.stream.str.replace(r"_seed\d+$", "", regex=True)
    df["seed"] = df.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    df["cfg"] = df.views + "|" + df.readout + "|" + df.base
    df.to_csv(base / "p11_rows.csv", index=False)
    out, lines = {}, []
    for part, d in df.groupby("part"):
        groups = {"near": P.OO_NEAR, "far": P.OO_FAR} if part == "openood" else {"fourood": P.FOUR}
        per_ds = d.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean()
        res = {"per_dataset": {f"{c}|{ds}": {k: float(v) for k, v in r.items()} for (c, ds), r in per_ds.iterrows()}}
        for g, sets in groups.items():
            dd = d[d.ds.isin(sets)]
            per_seed = dd.groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            ref = per_seed.loc[REF]
            gres = {}
            for c in per_seed.index.get_level_values(0).unique():
                v = per_seed.loc[c]
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_orders": int(len(v))}
                e["dAUROC_vs_ref"] = tci((v.AUROC - ref.AUROC).values)
                e["dFPR95_vs_ref"] = tci((v.FPR95 - ref.FPR95).values)
                gres[c] = e
            res[g] = gres
            lines.append(f"== {part} {g}: all configurations by FPR95 ==")
            for c, e in sorted(gres.items(), key=lambda kv: kv[1]["FPR95"]):
                dd_ = e["dFPR95_vs_ref"]
                lines.append(f"{c:44s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}   dFPR {dd_['mean']:+.2f} [{dd_['lo']:+.2f},{dd_['hi']:+.2f}]")
            lines.append(f"-- reference {REF}: {gres[REF]['AUROC']:.2f} / {gres[REF]['FPR95']:.2f}")
            lines.append("")
        out[part] = res
    # seed statistics of the main views (mean over the streams of a dataset)
    seeds = {}
    for part in ("openood", "fourood"):
        for f in sorted(glob.glob(str(base / part / "*.meta.json"))):
            m = json.loads(Path(f).read_text())
            ds = m["stream"].rsplit("_seed", 1)[0]
            for v in MAIN[0]:
                e = m["views"][v][f"q{Q_MAIN:g}"]
                seeds.setdefault(part, {}).setdefault(ds, {}).setdefault(v, []).append([e["seeds"], e["id_frac"], m["n_ood"], m["n"]])
    seed_summary = {}
    for part, dd in seeds.items():
        for ds, vv in dd.items():
            for v, L in vv.items():
                a = np.array(L, np.float64)
                seed_summary[f"{part}|{ds}|{v}"] = {"seeds": float(a[:, 0].mean()), "id_frac": float(np.nanmean(a[:, 1])),
                                                    "seeds_over_n_ood": float((a[:, 0] / a[:, 2]).mean()), "n_streams": int(len(a))}
    lines.append("== seeds of the main configuration (mean over streams): dataset|view  seeds  ID fraction  seeds/n_ood ==")
    for k, e in seed_summary.items():
        lines.append(f"{k:40s} {e['seeds']:8.0f} {e['id_frac']:7.3f} {e['seeds_over_n_ood']:7.3f}")
    out["_seeds"] = seed_summary
    out["_meta"] = {"utc": P.utc(), "readouts": READOUTS, "reference": REF, "main": ["+".join(MAIN[0]), MAIN[1], MAIN[2]],
                    "q": Q_MAIN, "k_g": KG, "lambda": 0.9, "n_streams": int(df.stream.nunique())}
    (base / "p11_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    t = "\n".join(lines)
    (base / "p11_table.txt").write_text(t + "\n")
    print(t)
    print("AN11F_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.aggregate:
        aggregate()
    else:
        run_part(a.part, a.worker, a.nworkers)
