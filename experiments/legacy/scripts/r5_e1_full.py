"""R5 Phase 1 / E1 (part 2): full-train kNN and Mahalanobis++ on DINOv2 L/14 (dev only, same 15 conditions).

Reference set of a dev split = every ImageNet-1K train image of its ID classes (features from r5_extra_feats.py
--what l14_train) minus the images of dev1 and dev2 (stream images and original shots), minus every R5 shot image
(all draws), minus every sealed sha256; byte-identical duplicates are kept once.
Scores (OOD direction): kNN = 1 - cos of the k-th nearest reference image (Sun et al. 2022), k in KS;
Mahalanobis++ (L2-normalised features, class means and one shared covariance of all reference residuals, shrinkage
lam in LAMS, as vins.r5.maha_pp); class-mean cosine (1 - max_c cos(x, mu_c)).
Alone: the raw score. x TINS: static conformal p against the draw's calibration shots (scored the same way), times
S_final of the draw's TINS run, as for every other baseline. Hyper-parameters are picked on dev1 (near FPR95).
Output: <R5>/<dev>/e1full/scores.npz (query scores) and <R5>/<dev>/e1full/draw<k>/<stream>_seed<s>.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.sealing import load_sealed  # noqa: E402

KS = (1, 5, 10, 50, 100, 200, 1000)
LAMS = (0.0, 0.001, 0.01, 0.1)
BASE = Path.home() / "vins_gonogo_20260925"


def dev_name():
    return "dev2" if C.WORK.name == "dev2" else "dev1"


def reference(id_classes):
    shards = sorted((r5.R5 / "features" / "in1k_train_l14").glob("shard*of*.pt"))
    shards = [p for p in shards if "_part" not in p.name]
    blobs = [torch.load(p, map_location="cpu") for p in shards]
    sid = np.array([s for b in blobs for s in b["sample_id"]])
    sha = np.array([h for b in blobs for h in b["sha256"]])
    feats = torch.cat([b["features"] for b in blobs])
    n_total = len(sid)
    excluded = set()
    for work in (BASE, BASE / "dev2"):
        excluded |= set(pd.read_parquet(work / "splits" / "samples.parquet").sha256)
    excluded |= set(pd.read_parquet(r5.R5 / "shots" / "draws.parquet").sha256)
    _, sealed_hashes, _ = load_sealed()
    excluded |= sealed_hashes
    wnid = np.array([s.split("/")[2] for s in sid])
    cls = {c["wnid"]: c["id_idx"] for c in id_classes}
    keep = np.array([w in cls for w in wnid]) & ~np.isin(sha, list(excluded))
    _, first = np.unique(sha, return_index=True)                  # byte-identical duplicates kept once
    uniq = np.zeros(len(sha), dtype=bool)
    uniq[first] = True
    keep &= uniq
    labels = np.array([cls[w] for w in wnid[keep]])
    info = {"n_train_total": int(n_total), "n_reference": int(keep.sum()), "n_classes": len(cls),
            "min_per_class": int(np.bincount(labels, minlength=len(cls)).min())}
    return feats[torch.from_numpy(keep)].float(), labels, info


@torch.no_grad()
def scores(ref, labels, queries, n_cls, step=131072):
    """ref stays on the CPU and is streamed to the GPU in chunks (about 2.5 GB of GPU memory in total)."""
    dev = torch.device("cuda")
    Q = torch.from_numpy(queries).float().to(dev)
    Q = Q / Q.norm(dim=1, keepdim=True)
    lab_all = torch.from_numpy(labels)
    kmax = max(KS)
    best = torch.full((len(Q), kmax), -2.0, device=dev)
    mu = torch.zeros(n_cls, ref.shape[1], dtype=torch.float64, device=dev)
    for lo in range(0, len(ref), step):
        R = ref[lo:lo + step].to(dev)
        lab = lab_all[lo:lo + step].to(dev)
        mu.index_add_(0, lab, R.double())
        for q0 in range(0, len(Q), 2048):
            sims = Q[q0:q0 + 2048] @ R.T
            k = min(kmax, sims.shape[1])
            cand = torch.cat([best[q0:q0 + 2048], sims.topk(k, dim=1).values], dim=1)
            best[q0:q0 + 2048] = cand.topk(kmax, dim=1).values
    top = best.double().cpu().numpy()
    out = {f"knnF{k}": 1.0 - top[:, k - 1] for k in KS}
    mu /= torch.bincount(lab_all, minlength=n_cls).double().to(dev)[:, None]
    mun = mu / mu.norm(dim=1, keepdim=True)
    out["protoF"] = 1.0 - (Q.double() @ mun.T).max(dim=1).values.cpu().numpy()
    cov = torch.zeros(ref.shape[1], ref.shape[1], dtype=torch.float64, device=dev)
    for lo in range(0, len(ref), step):
        res = ref[lo:lo + step].to(dev).double() - mu[lab_all[lo:lo + step].to(dev)]
        cov += res.T @ res
    cov = (cov / (len(ref) - n_cls)).cpu().numpy()
    mu = mu.cpu().numpy()
    Qn = Q.double().cpu().numpy()
    dim = cov.shape[0]
    for lam in LAMS:
        c = (1 - lam) * cov + lam * np.trace(cov) / dim * np.eye(dim)
        evals, evecs = np.linalg.eigh(c)
        evals = np.maximum(evals, 1e-10 * evals.max())
        Wm = evecs / np.sqrt(evals)
        muw = mu @ Wm
        mnorm = (muw ** 2).sum(axis=1)
        res = np.empty(len(Qn))
        for lo in range(0, len(Qn), 4096):
            qw = Qn[lo:lo + 4096] @ Wm
            res[lo:lo + 4096] = ((qw ** 2).sum(axis=1)[:, None] - 2 * qw @ muw.T + mnorm[None, :]).min(axis=1)
        out[f"mahaF{lam:g}"] = res
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", nargs="+", default=["0", "1", "2", "3", "4"])
    opts = parser.parse_args()
    start = time.time()
    from vins.tins_dev import import_tins
    from vins.metrics import measures as upstream

    import_tins()
    dev = dev_name()
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
    dev_f, _ = load_features(C.FEATURES_DIR / "dino_vitl14.pt", list(stream_ids))
    spec = importlib.util.spec_from_file_location("r5_eval", ROOT / "scripts" / "r5_eval.py")
    ev_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev_mod)
    cal_ids = {d: ev_mod.shot_table(d, id_classes, samples)[1] for d in opts.draws}
    all_cal = np.concatenate([cal_ids[d] for d in opts.draws])
    sh_f, _ = load_features(r5.R5 / "features" / "shots.dino_vitl14.pt", list(all_cal))
    queries = np.concatenate([dev_f.float().numpy(), sh_f.float().numpy()])
    ref, labels, info = reference(id_classes)
    sc = scores(ref, labels, queries, len(id_classes))
    out = r5.R5 / dev / "e1full"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", query_id=np.concatenate([stream_ids, all_cal]), **sc)
    row = {s: i for i, s in enumerate(stream_ids)}
    cal_row = {s: len(stream_ids) + i for i, s in enumerate(all_cal)}
    for d in opts.draws:
        (out / f"draw{d}").mkdir(exist_ok=True)
        cidx = np.array([cal_row[s] for s in cal_ids[d]])
        for stream in ("near", "far"):
            for seed in C.ORDER_SEEDS:
                z = np.load(r5.R5 / dev / "tins" / f"draw{d}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
                sid, is_ood, S = z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64)
                q = np.array([row[s] for s in sid])
                metrics = {}
                for key, val in sc.items():
                    p = r5.pval_high(val[cidx], val[q])
                    for name, s in ((f"v_{key}", -val[q]), (f"s_{key}", S * p)):
                        m = upstream(s[~is_ood], s[is_ood])
                        metrics[name] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
                (out / f"draw{d}" / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics}) + "\n")
    info["seconds"] = round(time.time() - start, 1)
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
