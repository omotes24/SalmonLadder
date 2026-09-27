"""Evaluate the frozen CLAVIS-M3 on one dev WORK against the pre-registered rule (iter3/prereg_iter3.json).

Usage: VINS_WORK=<work of the dev> python scripts/confirm_iter3.py --frozen F --prereg P --dev dev1|dev2 [--looks L]
The functions are imported from scripts/iter3_eval.py, whose sha256 must match the frozen file.
Every dev2 evaluation is appended to --looks before the verdict is printed.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402


def load_ev():
    spec = importlib.util.spec_from_file_location("iter3_eval", ROOT / "scripts" / "iter3_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--dev", choices=["dev1", "dev2"], required=True)
    parser.add_argument("--looks")
    opts = parser.parse_args()
    assert opts.dev != "dev2" or opts.looks, "dev2 looks must be logged"
    start = time.time()
    fb, pb = Path(opts.frozen).read_bytes(), Path(opts.prereg).read_bytes()
    frozen, prereg = json.loads(fb), json.loads(pb)
    frozen_sha, prereg_sha = hashlib.sha256(fb).hexdigest(), hashlib.sha256(pb).hexdigest()
    code_sha = hashlib.sha256((ROOT / "scripts" / "iter3_eval.py").read_bytes()).hexdigest()
    assert code_sha == frozen["code"]["sha256"], "iter3_eval.py changed after freezing"
    base = prereg["baseline"][opts.dev]
    cfg = frozen["config"]
    ev = load_ev()
    ev.import_tins()
    views = [ev.build_view(s) for s in cfg["views"]]
    m2_views = [ev.build_view(s) for s in ("dino.pt", "dino_vitl14.pt")]
    k_, a_, g_ = cfg["lp"].split(",")
    per_seed, mem = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
            sid, is_id, bidx = f.sample_id.values, (f.group == "ID").values, f.batch_index.values
            S = f.S_final.values.astype(np.float64)
            m2 = S.copy()
            for v in m2_views:
                p_t, _, _ = ev.view_pt(v, sid, bidx, "cand30_10")
                m2 = m2 * p_t
            pts, lps = [], []
            for i, v in enumerate(views):
                p_t, admit, _ = ev.view_pt(v, sid, bidx, cfg["rule"])
                p_lp = ev.lp_pvalues(v, v["feats"][v["row_of"].loc[sid].values], bidx, int(k_), float(a_), float(g_))
                pts.append(p_t)
                lps.append(p_lp)
                mem[f"v{i}|{stream}_seed{seed}"] = {"frac_ID_admitted": float(admit[is_id].mean()),
                                                   "frac_OOD_admitted": float(admit[~is_id].mean())}
            memory_part = np.prod(pts, axis=0)
            lp_part = np.prod(lps, axis=0)
            m3 = S * memory_part * lp_part
            entry[stream] = {"tins": measures(S[is_id], S[~is_id]), "clavism_m2": measures(m2[is_id], m2[~is_id]),
                             "candidate": measures(m3[is_id], m3[~is_id]),
                             "S*memory": measures((S * memory_part)[is_id], (S * memory_part)[~is_id]),
                             "S*lp": measures((S * lp_part)[is_id], (S * lp_part)[~is_id])}
        per_seed.append(entry)
    agg = aggregate(per_seed)

    def a(stream, key, metric="AUROC"):
        return 100 * agg[stream][key][metric]["mean"]

    near, far = a("near", "candidate"), a("far", "candidate")
    crit = {"near_needed": base["near_AUROC"] + 1.5, "far_needed": base["far_AUROC"] - 0.3}
    verdict = {"near": near, "far": far, "near_ok": near >= crit["near_needed"], "far_ok": far >= crit["far_needed"]}
    verdict["pass"] = bool(verdict["near_ok"] and verdict["far_ok"])
    record = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "dev": opts.dev, "frozen": opts.frozen, "frozen_sha256": frozen_sha, "prereg_sha256": prereg_sha,
              "criterion": crit, "verdict": verdict,
              "clavism_m2_reproduced": {"near": a("near", "clavism_m2"), "far": a("far", "clavism_m2")}}
    if opts.dev == "dev2":
        with open(opts.looks, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    out = C.WORK / "iter3" / f"confirm_{frozen['name']}_{opts.dev}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"record": record, "aggregate": agg, "memory": mem,
                                                  "seconds": round(time.time() - start, 1)}, indent=1) + "\n")
    rows = [("TINS S_final", "tins"), ("CLAVIS-M2", "clavism_m2"), (frozen["name"], "candidate"),
            ("S × memory p（proto ビュー）", "S*memory"), ("S × LP p", "S*lp")]
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
    for i, spec in enumerate(cfg["views"]):
        fid = np.mean([m["frac_ID_admitted"] for k, m in mem.items() if k.startswith(f"v{i}|")])
        fo = {st: np.mean([m["frac_OOD_admitted"] for k, m in mem.items() if k.startswith(f"v{i}|{st}")]) for st in ("near", "far")}
        lines.append(f"- {spec}: メモリに入った ID {100 * fid:.2f}%、near OOD {100 * fo['near']:.1f}%、far OOD {100 * fo['far']:.1f}%")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
