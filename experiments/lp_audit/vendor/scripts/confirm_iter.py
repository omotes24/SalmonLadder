"""Evaluate a frozen CLAVIS-M candidate on one dev WORK against the pre-registered stopping rule.

Usage: VINS_WORK=<work of the dev> python scripts/confirm_iter.py --frozen F --prereg P --dev dev1|dev2 [--looks L]
Every dev2 evaluation is appended to --looks (required for dev2) before the verdict is printed.
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
    frozen_bytes = Path(opts.frozen).read_bytes()
    frozen, frozen_sha = json.loads(frozen_bytes), hashlib.sha256(frozen_bytes).hexdigest()
    prereg_bytes = Path(opts.prereg).read_bytes()
    prereg, prereg_sha = json.loads(prereg_bytes), hashlib.sha256(prereg_bytes).hexdigest()
    base = prereg["baseline"][opts.dev]
    cfg = frozen["config"]
    import_tins()
    v = dev_views(m=cfg["m"])
    per_seed, mem = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
            is_id = (f.group == "ID").values
            s = f.S_final.values.astype(np.float64)
            fr = v["frame"].loc[f.sample_id.values]
            sf = v["feats"][v["row_of"].loc[f.sample_id.values].values]
            p_t, g, admit = score_stream(v, sf, fr.d.values, fr.p_all.values, f.batch_index.values,
                                         eps_admit=cfg["eps_admit"], m_mem=cfg["m_mem"], kind=cfg["kind"])
            clavis, cand = s * fr.p.values, s * p_t
            entry[stream] = {"tins": measures(s[is_id], s[~is_id]), "clavis": measures(clavis[is_id], clavis[~is_id]),
                             "candidate": measures(cand[is_id], cand[~is_id]), "visual_g": measures(-g[is_id], -g[~is_id])}
            mem[f"{stream}_seed{seed}"] = {"admitted": int(admit.sum()), "admitted_ID": int((admit & is_id).sum()),
                                          "frac_ID_admitted": float(admit[is_id].mean()),
                                          "frac_OOD_admitted": float(admit[~is_id].mean())}
        per_seed.append(entry)
    agg = aggregate(per_seed)

    def a(stream, key, metric="AUROC"):
        return 100 * agg[stream][key][metric]["mean"]

    near, far = a("near", "candidate"), a("far", "candidate")
    crit = {"near_needed": base["near_AUROC"] + 3.0, "far_needed": base["far_AUROC"] - 0.3}
    verdict = {"near": near, "far": far, "near_ok": near >= crit["near_needed"], "far_ok": far >= crit["far_needed"]}
    verdict["pass"] = bool(verdict["near_ok"] and verdict["far_ok"])
    record = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "dev": opts.dev, "frozen": opts.frozen, "frozen_sha256": frozen_sha, "prereg_sha256": prereg_sha,
              "criterion": crit, "verdict": verdict,
              "clavis_reproduced": {"near": a("near", "clavis"), "far": a("far", "clavis")}}
    if opts.dev == "dev2":
        with open(opts.looks, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    out = C.WORK / "iter" / f"confirm_{frozen['name']}_{opts.dev}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"record": record, "aggregate": agg, "memory": mem,
                                                  "seconds": round(time.time() - start, 1)}, indent=1) + "\n")
    rows = [("TINS S_final", "tins"), ("CLAVIS", "clavis"), (f"{frozen['name']}", "candidate"), ("視覚 g 単体", "visual_g")]
    lines = [f"# {frozen['name']} の評価（{opts.dev}）", "",
             f"- 凍結設定: `{Path(opts.frozen).name}`（sha256 `{frozen_sha[:16]}…`）、事前登録 sha256 `{prereg_sha[:16]}…`",
             f"- 判定: near {near:.2f}（必要 {crit['near_needed']:.2f}）、far {far:.2f}（必要 {crit['far_needed']:.2f}） → "
             f"**{'PASS' if verdict['pass'] else 'FAIL'}**", "",
             "| スコア | near AUROC | near FPR95 | far AUROC | far FPR95 |", "|---|---|---|---|---|"]
    for label, key in rows:
        sd = lambda st: 100 * agg[st][key]["AUROC"]["sd"]  # noqa: E731
        lines.append(f"| {label} | {a('near', key):.2f} ± {sd('near'):.2f} | {a('near', key, 'FPR95'):.2f} | "
                     f"{a('far', key):.2f} ± {sd('far'):.2f} | {a('far', key, 'FPR95'):.2f} |")
    fid = np.mean([m["frac_ID_admitted"] for k, m in mem.items()])
    lines += ["", f"- メモリに入った ID の割合（全ストリーム平均）: {100 * fid:.2f}%（上限の目安 {100 * cfg['eps_admit']:.0f}%）"]
    for stream in ("near", "far"):
        fo = np.mean([m["frac_OOD_admitted"] for k, m in mem.items() if k.startswith(stream)])
        lines.append(f"- {stream} の OOD がメモリに入った割合: {100 * fo:.1f}%")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
