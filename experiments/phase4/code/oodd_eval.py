"""Original-method baseline (Exp 2): MCM+OODD exactly as the official OODD repository's MCM variant
(OODD/MCM/eval_ood_detection.py): a dictionary of the 2,048 lowest-MCM features seen so far (current batch
inserted before scoring), score = MCM - 5th largest cosine to the dictionary; batch 64 (official) and 256.
Same CLIP ViT-B/16 features, ID names and stream orders as the confirmatory evaluation."""
import json
import os

import numpy as np
import pandas as pd
import torch

from bank_data import Features, pos_text
from common import RESULTS, utc
from metrics_p4 import metrics

OUT = RESULTS / "oodd"


def oodd_scores(clip, mcm, batch=64, qsize=2048, k=5, device="cuda"):
    X = torch.as_tensor(clip, device=device)
    q_idx = np.zeros(0, np.int64)
    out = np.empty(len(mcm))
    for lo in range(0, len(mcm), batch):
        rows = np.arange(lo, min(lo + batch, len(mcm)))
        cand = np.r_[q_idx, rows]
        order = np.argsort(mcm[cand], kind="stable")[:qsize]       # keep the most OOD-like (lowest MCM)
        q_idx = cand[order]
        sims = X[torch.as_tensor(q_idx, device=device)] @ X[torch.as_tensor(rows, device=device)].T
        kk = min(k, len(q_idx))
        kth = torch.topk(sims, kk, dim=0).values[-1].cpu().numpy()
        out[rows] = mcm[rows] - kth
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = Features()
    rows = []
    for f in sorted((RESULTS / "eval").glob("*_scores.npz")):
        name = f.name.replace("_scores.npz", "")
        bank, s, d, seed = name.split("_")
        z = np.load(f, allow_pickle=True)
        ids, flag = list(z["sample_id"]), z["is_ood"].astype(bool)
        pt = pos_text(bank, int(s[1:]))
        clip = F.get("CLIP", ids)
        mcm = torch.softmax(torch.as_tensor(clip @ pt.T, dtype=torch.float64), dim=1).max(1).values.numpy()
        for b in (64, 256):
            sc = oodd_scores(clip, mcm, batch=b)
            rows.append({"bank": bank, "split": int(s[1:]), "draw": int(d[1:]), "seed": int(seed[4:]), "family": f"MCM+OODD_b{b}",
                         **metrics(sc, flag)})
        rows.append({"bank": bank, "split": int(s[1:]), "draw": int(d[1:]), "seed": int(seed[4:]), "family": "MCM",
                     **metrics(np.log(mcm), flag)})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "oodd.parquet", index=False)
    print(df.groupby(["bank", "family"])[["AUROC", "FPR95"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
