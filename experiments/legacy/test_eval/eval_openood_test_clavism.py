"""SECOND evaluation on the sealed OpenOOD v1.5 ImageNet-1K test: the frozen CLAVIS-M_c1 (run exactly once).

Pre-registered in test_eval/prereg_test2.json before any CLAVIS-M score on the test is computed.
  TINS S_final : Codex's unchanged-upstream scores (stream seed 123, batch 256), as in the first evaluation
  features     : test_eval/results/features.pt stored by the first evaluation (no test image is re-read)
  CLAVIS-M_c1  : vins/clavism.py with iter/frozen_c1.json; the memory is reset for every ID+OOD stream (like TINS's bank)
Reported for TINS, CLAVIS, CLAVIS-M_c1 and the visual score g alone; nothing else is evaluated.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import score_stream, views_from_arrays  # noqa: E402
from vins.tins_dev import build_order, import_tins  # noqa: E402

HERE = Path(__file__).resolve().parent
STREAMS = {"nearood": ["ssb_hard", "ninco"], "farood": ["inaturalist", "textures", "openimageo"]}
BATCH = 256
STREAM_SEED = 123


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codex_fpr95(id_scores, ood_scores):
    iid, ood = -np.asarray(id_scores, float), -np.asarray(ood_scores, float)
    threshold = np.sort(iid)[math.ceil(0.95 * len(iid)) - 1]
    return float(np.mean(ood <= threshold))


def metrics(t, id_s, ood_s):
    auroc, _, fpr = t.get_measures(np.asarray(id_s, float), np.asarray(ood_s, float))
    auroc_sk = roc_auc_score(np.r_[np.ones(len(id_s)), np.zeros(len(ood_s))], np.r_[id_s, ood_s])
    return {"AUROC": float(auroc), "AUROC_sklearn": float(auroc_sk), "FPR95_upstream": float(fpr),
            "FPR95_codex": codex_fpr95(id_s, ood_s)}


def main():
    start = time.time()
    out = C.WORK / "test_eval" / "results_clavism"
    if (out / "metrics.json").exists():
        raise SystemExit("already evaluated once; refusing to run again")
    prereg = json.loads((HERE / "prereg_test2.json").read_text())
    prereg_sha = sha256(HERE / "prereg_test2.json")
    frozen_path = C.WORK / "iter" / "frozen_c1.json"
    assert sha256(frozen_path) == prereg["frozen_candidate_sha256"], "frozen candidate changed"
    feat_path = C.WORK / "test_eval" / "results" / "features.pt"
    assert sha256(feat_path) == prereg["features_sha256"], "stored features changed"
    looks = [json.loads(line) for line in (C.WORK / "iter" / "dev2_looks.jsonl").read_text().splitlines() if line.strip()]
    mine = [r for r in looks if r["frozen_sha256"] == prereg["frozen_candidate_sha256"]]
    assert mine and mine[-1]["verdict"]["pass"], "dev2 confirmation did not pass"
    cfg = json.loads(frozen_path.read_text())["config"]
    out.mkdir(parents=True, exist_ok=True)

    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    scores = {}
    for group, names in STREAMS.items():
        for name in names:
            z = np.load(C.CODEX_TINS / f"scores_openood_{group}_{name}.npz", allow_pickle=True)
            scores[name] = {"sample_id": z["sample_id"].astype(str), "is_ood": z["is_ood"], "S": z["ID_score"]}
    test_ids = sorted({s for v in scores.values() for s in v["sample_id"]})
    row_of = {s: i for i, s in enumerate(test_ids)}

    blob = torch.load(feat_path, map_location="cpu")
    n_sup, n_cal = 1000 * C.N_SUPPORT, 1000 * C.N_CALIB
    paths = blob["paths"]
    assert len(paths) == n_sup + n_cal + len(test_ids)
    assert all(paths[n_sup + n_cal + i] == manifest[s]["path"] for i, s in enumerate(test_ids)), "feature/path mismatch"
    dino = blob["dino"].numpy().astype(np.float32)
    support = dino[:n_sup].reshape(1000, C.N_SUPPORT, -1)
    queries = dino[n_sup:]
    is_cal = np.zeros(len(queries), dtype=bool)
    is_cal[:n_cal] = True
    tick = time.time()
    v = views_from_arrays(support, queries, blob["cand"], is_cal, m=cfg["m"])
    view_seconds = round(time.time() - tick, 1)
    t = import_tins()

    results, memory = {}, {}
    for group, names in STREAMS.items():
        for name in names:
            sc = scores[name]
            sid, is_ood, s = sc["sample_id"], sc["is_ood"].astype(int), sc["S"].astype(np.float64)
            prefix = {x.rsplit("_", 1)[0] for x, o in zip(sid, is_ood) if o == 1}
            assert len(prefix) == 1
            prefix = prefix.pop()
            order = build_order(int((is_ood == 0).sum()), int((is_ood == 1).sum()), STREAM_SEED)
            expected = [f"imagenet_{i:06d}" if o == 0 else f"{prefix}_{i:06d}" for o, i in order]
            assert list(sid) == expected, f"{name}: scores are not in upstream stream order"
            q = n_cal + np.array([row_of[x] for x in sid])
            bidx = np.arange(len(sid)) // BATCH
            tick = time.time()
            p_t, g, admit = score_stream(v, queries[q], v["d"][q], v["p_all"][q], bidx,
                                         eps_admit=cfg["eps_admit"], m_mem=cfg["m_mem"], kind=cfg["kind"])
            is_id = is_ood == 0
            clavis, clavism = s * v["p"][q], s * p_t
            results[name] = {"n_ID": int(is_id.sum()), "n_OOD": int((~is_id).sum()),
                             "tins": metrics(t, s[is_id], s[~is_id]),
                             "clavis": metrics(t, clavis[is_id], clavis[~is_id]),
                             "clavism": metrics(t, clavism[is_id], clavism[~is_id]),
                             "visual_g": metrics(t, -g[is_id], -g[~is_id])}
            memory[name] = {"admitted": int(admit.sum()), "frac_ID_admitted": float(admit[is_id].mean()),
                            "frac_OOD_admitted": float(admit[~is_id].mean()), "seconds": round(time.time() - tick, 1)}
            print(json.dumps({name: memory[name]}), flush=True)
    methods = ("tins", "clavis", "clavism", "visual_g")
    summary = {g_: {m: {k: float(np.mean([results[n][m][k] for n in names])) for k in ("AUROC", "FPR95_codex", "FPR95_upstream")}
                    for m in methods} for g_, names in STREAMS.items()}
    first = json.loads((C.WORK / "test_eval" / "results" / "metrics.json").read_text())
    repro = {n: abs(results[n]["clavis"]["AUROC"] - first["results"][n]["fused"]["AUROC"]) for n in results}
    blob_out = {"prereg_sha256": prereg_sha, "frozen": cfg, "results": results, "means": summary, "memory": memory,
                "clavis_reproduction_max_abs_diff_vs_first_eval": max(repro.values()),
                "view_seconds": view_seconds, "seconds": round(time.time() - start, 1)}
    (out / "metrics.json").write_text(json.dumps(blob_out, indent=1) + "\n")

    labels = {"tins": "TINS", "clavis": "CLAVIS", "clavism": "CLAVIS-M", "visual_g": "視覚 g 単体"}
    L = ["# OpenOOD ImageNet-1K test：CLAVIS-M（test の 2 回目の使用、1 回だけの評価）\n",
         f"- 事前登録: `test_eval/prereg_test2.json`（sha256 `{prereg_sha[:16]}…`）。dev1・dev2 で停止基準を満たした凍結候補 `iter/frozen_c1.json`。",
         "- TINS は Codex による upstream 無改変の再現スコア（ストリーム順・バッチ 256）。特徴は 1 回目の評価で保存したものを再利用。",
         f"- CLAVIS の再現差（1 回目の評価との AUROC の最大差）: {blob_out['clavis_reproduction_max_abs_diff_vs_first_eval']:.2e}",
         "- AUROC / FPR95（%、FPR95 は Codex の規約）。\n",
         "| データセット | " + " | ".join(labels[m] for m in methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for g_, names in STREAMS.items():
        for n in names:
            r = results[n]
            L.append(f"| {n} | " + " | ".join(f"{100 * r[m]['AUROC']:.2f} / {100 * r[m]['FPR95_codex']:.2f}" for m in methods) + " |")
    for g_ in STREAMS:
        L.append(f"| **{g_} 平均** | " + " | ".join(
            f"**{100 * summary[g_][m]['AUROC']:.2f} / {100 * summary[g_][m]['FPR95_codex']:.2f}**" for m in methods) + " |")
    L += ["", "| データセット | メモリ件数 | ID の採用率 | OOD の採用率 |", "|---|---|---|---|"]
    for n, mstat in memory.items():
        L.append(f"| {n} | {mstat['admitted']} | {100 * mstat['frac_ID_admitted']:.2f}% | {100 * mstat['frac_OOD_admitted']:.1f}% |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
