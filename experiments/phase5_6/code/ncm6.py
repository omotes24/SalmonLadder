"""Phase 6 sanity check of the features (train shots only; no stream image, no OOD label):
nearest-class-mean accuracy of the 4 calibration shots per ID class, prototypes = mean of the 12 support shots
(cosine), for every draw and view. A broken read-out (wrong token, wrong preprocessing) would show up here."""
import json
import sys

import numpy as np
import torch

import dev5

dev = sys.argv[1] if len(sys.argv) > 1 else "dev1"
D = dev5.load(dev)
views = [v for v in ("B14", "L14", "D3B", "D3L", "D3Bs", "D3Ls") if v in D["feats"]]
rows = np.zeros(0, dtype=int)
out = {}
for v in views:
    acc = []
    for d in ("0", "1", "2", "3", "4"):
        sup, cal, _ = dev5.raw_view(D, d, v, rows)
        sup, cal = torch.as_tensor(sup).cuda(), torch.as_tensor(cal).cuda()
        mu = torch.nn.functional.normalize(sup.mean(1), dim=1)
        pred = (torch.nn.functional.normalize(cal, dim=1) @ mu.T).argmax(1).cpu().numpy()
        lab = np.repeat(np.arange(D["n_id"]), 4)
        acc.append(100 * float((pred == lab).mean()))
    out[v] = {"mean": round(float(np.mean(acc)), 2), "per_draw": [round(a, 2) for a in acc], "dim": int(sup.shape[-1])}
    print(dev, v, out[v], flush=True)
open(f"/home/omote/reprise_p5_20261002/results/ncm6_{dev}.json", "w").write(json.dumps(out, indent=1) + "\n")
