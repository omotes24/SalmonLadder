"""Evaluate a frozen iteration-2 candidate on one dev WORK against the pre-registered rule (iter2/prereg_iter2.json).

Usage: VINS_WORK=<work of the dev> python scripts/confirm_iter2.py --frozen F --prereg P --dev dev1|dev2 [--looks L]
Every dev2 evaluation is appended to --looks (required for dev2) before the verdict is printed.
TINS side: the unchanged-upstream runs of the dev (runs/main, 3 order seeds).
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import dev_views, score_stream  # noqa: E402
from vins.iter2 import score_stream_cand, views_any  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--dev", choices=["dev1", "dev2"], required=True)
    parser.add_argument("--looks")
    opts = parser.parse_args()
    assert opts.dev != "dev2" or opts.looks, "dev2 looks must be logged"
    start = time.time()
    frozen_bytes, prereg_bytes = Path(opts.frozen).read_bytes(), Path(opts.prereg).read_bytes()
    frozen, prereg = json.loads(frozen_bytes), json.loads(prereg_bytes)
    frozen_sha, prereg_sha = hashlib.sha256(frozen_bytes).hexdigest(), hashlib.sha256(prereg_bytes).hexdigest()
    base = prereg["baseline"][opts.dev]
    cfg = frozen["config"]
    import_tins()
    v0 = dev_views(m=2)
    views = [v0 if f == "dino.pt" else views_any(f, m=cfg["m"]) for f in cfg["dino_files"]]
    per_seed, mem = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
            sid, is_id, bidx = f.sample_id.values, (f.group == "ID").values, f.batch_index.values
            s = f.S_final.values.astype(np.float64)
            fr0 = v0["frame"].loc[sid]
            p0, _, _ = score_stream(v0, v0["feats"][v0["row_of"].loc[sid].values], fr0.d.values, fr0.p_all.values, bidx)
            c1 = s * p0
            prod = np.ones(len(s))
            entry[stream] = {"tins": measures(s[is_id], s[~is_id]), "clavism_m": measures(c1[is_id], c1[~is_id])}
            for vi, (f_name, v) in enumerate(zip(cfg["dino_files"], views)):
                fr = v["frame"].loc[sid]
                p_t, g, admit, cand = score_stream_cand(v, v["feats"][v["row_of"].loc[sid].values], fr.d.values,
                                                        fr.p_all.values, bidx, eps_cand=cfg["eps_cand"],
                                                        eps_admit=cfg["eps_admit"], m_mem=cfg["m_mem"], kind=cfg["kind"])
                prod *= p_t
                sv = s * p_t
                entry[stream][f"view{vi}"] = measures(sv[is_id], sv[~is_id])
                entry[stream][f"visual_g{vi}"] = measures(-g[is_id], -g[~is_id])
                mem[f"view{vi}|{stream}_seed{seed}"] = {"frac_ID_admitted": float(admit[is_id].mean()),
                                                       "frac_OOD_admitted": float(admit[~is_id].mean()),
                                                       "frac_ID_candidates": float(cand[is_id].mean()),
                                                       "frac_OOD_candidates": float(cand[~is_id].mean())}
            cnd = s * prod
            entry[stream]["candidate"] = measures(cnd[is_id], cnd[~is_id])
            entry[stream]["visual_prod"] = measures(prod[is_id], prod[~is_id])
        per_seed.append(entry)
    agg = aggregate(per_seed)

    def a(stream, key, metric="AUROC"):
        return 100 * agg[stream][key][metric]["mean"]

    near, far = a("near", "candidate"), a("far", "candidate")
    crit = {"near_needed": base["near_AUROC"] + 2.0, "far_needed": base["far_AUROC"] - 0.3}
    verdict = {"near": near, "far": far, "near_ok": near >= crit["near_needed"], "far_ok": far >= crit["far_needed"]}
    verdict["pass"] = bool(verdict["near_ok"] and verdict["far_ok"])
    record = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "dev": opts.dev, "frozen": opts.frozen, "frozen_sha256": frozen_sha, "prereg_sha256": prereg_sha,
              "criterion": crit, "verdict": verdict,
              "clavism_m_reproduced": {"near": a("near", "clavism_m"), "far": a("far", "clavism_m")}}
    if opts.dev == "dev2":
        with open(opts.looks, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    out = C.WORK / "iter2" / f"confirm_{frozen['name']}_{opts.dev}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"record": record, "aggregate": agg, "memory": mem,
                                                  "seconds": round(time.time() - start, 1)}, indent=1) + "\n")
    rows = [("TINS S_final", "tins"), ("CLAVIS-M c1", "clavism_m"), (frozen["name"], "candidate"), ("視覚 p の積のみ", "visual_prod")]
    rows += [(f"S_final × p_t（{f}）", f"view{i}") for i, f in enumerate(cfg["dino_files"])]
    lines = [f"# {frozen['name']} の評価（{opts.dev}）", "",
             f"- 凍結設定: `{Path(opts.frozen).name}`（sha256 `{frozen_sha[:16]}…`）、事前登録 sha256 `{prereg_sha[:16]}…`",
             f"- 判定: near {near:.2f}（必要 {crit['near_needed']:.2f}）、far {far:.2f}（必要 {crit['far_needed']:.2f}） → "
             f"**{'PASS' if verdict['pass'] else 'FAIL'}**", "",
             "| スコア | near AUROC | near FPR95 | far AUROC | far FPR95 |", "|---|---|---|---|---|"]
    for label, key in rows:
        sd = lambda st: 100 * agg[st][key]["AUROC"]["sd"]  # noqa: E731
        lines.append(f"| {label} | {a('near', key):.2f} ± {sd('near'):.2f} | {a('near', key, 'FPR95'):.2f} | "
                     f"{a('far', key):.2f} ± {sd('far'):.2f} | {a('far', key, 'FPR95'):.2f} |")
    lines.append("")
    for vi, f_name in enumerate(cfg["dino_files"]):
        fid = np.mean([m["frac_ID_admitted"] for k, m in mem.items() if k.startswith(f"view{vi}|")])
        fo = {st: np.mean([m["frac_OOD_admitted"] for k, m in mem.items() if k.startswith(f"view{vi}|{st}")])
              for st in ("near", "far")}
        lines.append(f"- {f_name}: メモリに入った ID {100 * fid:.2f}%（目安 {100 * cfg['eps_admit']:.0f}%）、"
                     f"near OOD {100 * fo['near']:.1f}%、far OOD {100 * fo['far']:.1f}%")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
