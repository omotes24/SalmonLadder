"""ONE-TIME evaluation on the sealed OpenOOD ImageNet-1K test (run only after the dev2 primary endpoint passed).

Frozen method (test_eval/prereg_test.json): fused(x) = S_final(x) * p_d(x)
  S_final : TINS scores of Codex's unchanged-upstream reproduction (scores_openood_*.npz, 1000 ID classes)
  d(x)    : DINOv2 ViT-B/14 CLS view, m = 2, K(x) = CLIP zero-shot top-5 over the 1000 labels (TINS template),
            support = first 12 of TINS's 16 shots per class, LOO med/MAD shrunk with n0 = 12
  p_d(x)  : conformal p-value against the 4,000 calibration shots (last 4 of the 16 per class)
Compared with TINS alone and d(x) alone, per OOD dataset and near/far means.
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
from vins.dview import d_scores, loo_stats  # noqa: E402
from vins.features import dino_encode_fn, dino_transform, encode, load_dino  # noqa: E402
from vins.tins_dev import get_logger, import_tins, load_clip, make_args  # noqa: E402

HERE = Path(__file__).resolve().parent
STREAMS = {"nearood": ["ssb_hard", "ninco"], "farood": ["inaturalist", "textures", "openimageo"]}


class Open:
    """Test images are read on purpose here: this script is the single, pre-registered test evaluation."""

    @staticmethod
    def check(path):
        return path


def pvalue(cal, d):
    cal = np.sort(cal)
    return (1.0 + len(cal) - np.searchsorted(cal, d, side="left")) / (len(cal) + 1.0)


def codex_fpr95(id_scores, ood_scores):
    """FPR at 95% ID TPR, Codex's convention (threshold on OOD-direction scores, ties counted as detected-ID)."""
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
    prereg_bytes = (HERE / "prereg_test.json").read_bytes()
    prereg_sha = hashlib.sha256(prereg_bytes).hexdigest()
    confirm = json.loads((C.WORK / "dev2" / "reports" / "confirm" / "metrics.json").read_text())
    if not confirm["primary"]["pass"]:
        raise SystemExit("dev2 primary endpoint did not pass; the test evaluation is not run")
    out = C.WORK / "test_eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    codex = C.CODEX_TINS
    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    scores = {}
    for group, names in STREAMS.items():
        for name in names:
            z = np.load(codex / f"scores_openood_{group}_{name}.npz", allow_pickle=True)
            scores[name] = {"sample_id": z["sample_id"].astype(str), "is_ood": z["is_ood"], "S": z["ID_score"]}
    test_ids = sorted({s for v in scores.values() for s in v["sample_id"]})
    test_paths = [manifest[s]["path"] for s in test_ids]

    # 16-shot list for all 1000 classes: first 12 support, last 4 calibration
    shots = [line.split() for line in C.PROTO_LIST_SRC.read_text().splitlines() if line.strip()]
    by_label = {}
    for rel, label in shots:
        by_label.setdefault(int(label), []).append(str(C.IMAGENET_ROOT / rel))
    sup_paths = [p for c in range(1000) for p in by_label[c][:C.N_SUPPORT]]
    cal_paths = [p for c in range(1000) for p in by_label[c][C.N_SUPPORT:]]
    all_paths = sup_paths + cal_paths + test_paths
    timings = {}

    # CLIP (TINS loader) for zero-shot candidates over the 1000 labels
    t = import_tins()
    args = make_args(t, out / "tins_cache", "vins_test_eval")
    log = get_logger(out / "eval.log")
    net, preprocess = load_clip(t, args)
    labels = [str(x) for x in t.get_test_labels(args, None)]
    pos = t.encode_texts(net, [args.pos_prompt.format(l) for l in labels], batch_size=args.text_batch_size,
                         device="cuda", desc="Encoding positive labels").cuda()
    tick = time.time()
    clip_q = encode(cal_paths + test_paths, preprocess, net.encode_image, Open, batch_size=256, num_workers=8, desc="clip")
    timings["clip_s"] = round(time.time() - tick, 1)
    with torch.no_grad():
        cand = (clip_q.cuda() @ pos.T).topk(C.K_TOP, dim=1).indices.cpu().numpy()
    del net
    torch.cuda.empty_cache()
    log.debug("zero-shot candidates computed")

    # DINOv2 view
    tick = time.time()
    model = load_dino()
    dino = encode(all_paths, dino_transform(), dino_encode_fn(model), Open, batch_size=128, num_workers=8, desc="dino")
    timings["dino_s"] = round(time.time() - tick, 1)
    torch.save({"paths": all_paths, "dino": dino, "clip_q": clip_q, "cand": cand}, out / "features.pt")
    dino = dino.numpy().astype(np.float64)
    n_sup = len(sup_paths)
    support = dino[:n_sup].reshape(1000, C.N_SUPPORT, -1)
    queries = dino[n_sup:]
    stats = loo_stats(support, 2, C.N0, C.MAD_SCALE)
    d, _, _, _ = d_scores(queries, cand, support, stats, 2)
    d_cal, d_test = d[:len(cal_paths)], d[len(cal_paths):]
    d_of = dict(zip(test_ids, d_test))

    results = {}
    for name, v in scores.items():
        s = v["S"].astype(np.float64)
        dd = np.array([d_of[x] for x in v["sample_id"]])
        fused = s * pvalue(d_cal, dd)
        is_id = v["is_ood"] == 0
        results[name] = {"n_ID": int(is_id.sum()), "n_OOD": int((~is_id).sum()),
                         "tins": metrics(t, s[is_id], s[~is_id]),
                         "d_only": metrics(t, -dd[is_id], -dd[~is_id]),
                         "fused": metrics(t, fused[is_id], fused[~is_id])}
    summary = {}
    for group, names in STREAMS.items():
        summary[group] = {m: {k: float(np.mean([results[n][m][k] for n in names])) for k in ("AUROC", "FPR95_codex", "FPR95_upstream")}
                          for m in ("tins", "d_only", "fused")}
    blob = {"prereg_sha256": prereg_sha, "dev2_confirm_primary": confirm["primary"], "results": results,
            "means": summary, "timings": timings, "seconds": round(time.time() - start, 1),
            "codex_point_results_check": json.loads((codex / "point_results.json").read_text())["means"]}
    (out / "metrics.json").write_text(json.dumps(blob, indent=1) + "\n")

    L = ["# OpenOOD ImageNet-1K test（1 回だけの評価）\n",
         f"- 事前登録: `test_eval/prereg_test.json`（sha256 `{prereg_sha[:16]}…`）。dev2 の主要評価項目は PASS。",
         "- TINS: Codex による upstream 無改変の再現スコア（1000 クラス）。d(x): DINOv2・m=2・zs5（1000 クラス、支持 12 枚／較正 4 枚）。",
         "- AUROC / FPR95（%）。FPR95 は Codex の再現と同じ規約（閾値の同値を含む）。\n",
         "| データセット | TINS | d(x) 単体 | 融合 S_final × p_d |", "|---|---|---|---|"]
    for group, names in STREAMS.items():
        for n in names:
            r = results[n]
            L.append(f"| {n}（{group}） | " + " | ".join(
                f"{100 * r[m]['AUROC']:.2f} / {100 * r[m]['FPR95_codex']:.2f}" for m in ("tins", "d_only", "fused")) + " |")
    for group in STREAMS:
        L.append(f"| **{group} 平均** | " + " | ".join(
            f"**{100 * summary[group][m]['AUROC']:.2f} / {100 * summary[group][m]['FPR95_codex']:.2f}**"
            for m in ("tins", "d_only", "fused")) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"means": summary, "seconds": blob["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
