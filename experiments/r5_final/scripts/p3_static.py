"""Phase 3: static baselines on test (raw OOD-direction distances; p-values are taken in p3_eval.py).

Row space per part = [4,000 calibration shots; evaluated images in p3_eval order]. Hyper-parameters come from the
Phase 3 pre-registration (chosen on dev1):
  knn16_<view> / maha16_<view> : 16-shot kNN (k) and Mahalanobis++ (lam) of the 12 support shots, views B14 and L14
  knn16l_L14 / maha16l_L14     : the L14-alone selections (k_l14, lam_l14)
  knnF / mahaF / protoF        : full ImageNet-1K train on DINOv2 L/14 (r5_extra_feats.py shards), minus the 16k
                                 shots and every sealed sha256, byte duplicates once; kNN k, Mahalanobis++ lam, class means
Output: <R5>/phase3/static_<part>.npz
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.sealing import load_sealed  # noqa: E402

P3 = r5.R5 / "phase3"


@torch.no_grad()
def knn_gpu(ref, q, k, step=131072, qstep=2048):
    dev = torch.device("cuda")
    Q = torch.from_numpy(q).float().to(dev)
    Q = Q / Q.norm(dim=1, keepdim=True)
    best = torch.full((len(Q), k), -2.0, device=dev)
    for lo in range(0, len(ref), step):
        R = torch.from_numpy(np.ascontiguousarray(ref[lo:lo + step])).float().to(dev)
        R = R / R.norm(dim=1, keepdim=True)
        for q0 in range(0, len(Q), qstep):
            sims = Q[q0:q0 + qstep] @ R.T
            cand = torch.cat([best[q0:q0 + qstep], sims.topk(min(k, sims.shape[1]), dim=1).values], dim=1)
            best[q0:q0 + qstep] = cand.topk(k, dim=1).values
    return 1.0 - best[:, k - 1].double().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["openood", "fourood"], required=True)
    opts = parser.parse_args()
    start = time.time()
    blob = (P3 / "prereg_phase3.json").read_bytes()
    pr = json.loads(blob)
    b = pr["baselines"]
    E = __import__("importlib").import_module("importlib.util")
    spec = E.spec_from_file_location("p3_eval", ROOT / "scripts" / "p3_eval.py")
    P = E.module_from_spec(spec)
    spec.loader.exec_module(P)
    D = P.load_part(opts.part)
    out = {}
    for name in ("B14", "L14"):
        sup = D["shots"][name][:12000]
        q = np.concatenate([D["shots"][name][12000:16000], D["ev"][name]])
        out[f"knn16_{name}"] = knn_gpu(sup, q, b["knn_k"])
        out[f"maha16_{name}"] = r5.maha_pp(sup.reshape(1000, 12, -1), q, (b["maha_lam"],))[b["maha_lam"]]
        if name == "L14":
            out["knn16l_L14"] = knn_gpu(sup, q, b["knn_k_l14"])
            out["maha16l_L14"] = r5.maha_pp(sup.reshape(1000, 12, -1), q, (b["maha_lam_l14"],))[b["maha_lam_l14"]]
    # full-train reference on L/14 (all 1000 classes)
    shards = sorted(p for p in (r5.R5 / "features" / "in1k_train_l14").glob("shard*of*.pt") if "_part" not in p.name)
    blobs = [torch.load(p, map_location="cpu") for p in shards]
    sid = np.array([s for x in blobs for s in x["sample_id"]])
    sha = np.array([h for x in blobs for h in x["sha256"]])
    feats = torch.cat([x["features"] for x in blobs]).float().numpy()
    bb = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    shot_names = {Path(p).parent.name + "/" + Path(p).name for p in bb["paths"][:16000]}
    _, sealed, _ = load_sealed()
    wn = np.array([s.split("/")[2] for s in sid])
    keep = np.array([f"{w}/{s.split('/')[-1]}" not in shot_names for w, s in zip(wn, sid)]) & ~np.isin(sha, list(sealed))
    _, first = np.unique(sha, return_index=True)
    uniq = np.zeros(len(sha), dtype=bool)
    uniq[first] = True
    keep &= uniq
    wnids = sorted(set(wn))
    cls = np.array([wnids.index(w) for w in wn[keep]]) if len(wnids) == 1000 else None
    ref = feats[keep]
    q = np.concatenate([D["shots"]["L14"][12000:16000], D["ev"]["L14"]])
    out["knnF"] = knn_gpu(ref, q, b["knnF_k"])
    R = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    mu = np.zeros((1000, R.shape[1]))
    np.add.at(mu, cls, R)
    mu /= np.bincount(cls, minlength=1000)[:, None]
    mun = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    Qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    out["protoF"] = 1.0 - (Qn @ mun.T).max(axis=1)
    res = R - mu[cls]
    cov = res.T @ res / (len(R) - 1000)
    dim = cov.shape[0]
    lam = b["mahaF_lam"]
    c = (1 - lam) * cov + lam * np.trace(cov) / dim * np.eye(dim)
    ev_, evec = np.linalg.eigh(c)
    ev_ = np.maximum(ev_, 1e-10 * ev_.max())
    Wm = evec / np.sqrt(ev_)
    muw = mu @ Wm
    mn = (muw ** 2).sum(axis=1)
    md = np.empty(len(Qn))
    for lo in range(0, len(Qn), 4096):
        qw = Qn[lo:lo + 4096] @ Wm
        md[lo:lo + 4096] = ((qw ** 2).sum(axis=1)[:, None] - 2 * qw @ muw.T + mn[None, :]).min(axis=1)
    out["mahaF"] = md
    np.savez_compressed(P3 / f"static_{opts.part}.npz", **out)
    print(json.dumps({"part": opts.part, "n_reference": int(keep.sum()), "prereg_sha256": hashlib.sha256(blob).hexdigest(),
                      "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
