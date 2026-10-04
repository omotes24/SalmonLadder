"""Feature-quality indicator per view (labels of ID images only; no OOD label): 12-shot nearest-class-mean accuracy
on the ID evaluation images. usage: ncm7.py u1 VIEW,VIEW,...  |  ncm7.py ds"""
import json
import sys

import numpy as np
import torch

import data7
from p7common import RESULTS


def ncm(P, view):
    sup = torch.as_tensor(P.feat(view, P.sup_ids)).cuda().reshape(P.C, 12, -1)
    mu = torch.nn.functional.normalize(sup.mean(1), dim=1)
    x = torch.as_tensor(P.feat(view, P.id_ids)).cuda()
    pred = (x @ mu.T).argmax(1).cpu().numpy()
    names = {}
    cl = P.cls(P.id_ids)
    sup_cls = P.cls(P.sup_ids)[::12]
    rank = {c: i for i, c in enumerate(sup_cls)}
    lab = np.array([rank[c] for c in cl])
    return 100 * float((pred == lab).mean())


if __name__ == "__main__":
    which = sys.argv[1]
    out = {}
    if which == "u1":
        for v in sys.argv[2].split(","):
            acc = [ncm(data7.u1(s, 0), v) for s in range(1, 6)]
            out[v] = {"mean": float(np.mean(acc)), "per_split": acc}
            print("U1", v, round(out[v]["mean"], 2), flush=True)
        f = RESULTS / "ncm_u1.json"
        old = json.loads(f.read_text()) if f.exists() else {}
        old.update(out)
        f.write_text(json.dumps(old, indent=1) + "\n")
    else:
        for ds in ("cub", "cifar100", "places365", "inr", "insk"):
            for v in ("B14", "L14", "CLIP"):
                acc = [ncm(data7.ds(ds, s), v) for s in range(1, 6)]
                out[f"{ds}|{v}"] = {"mean": float(np.mean(acc)), "per_split": acc}
                print(ds, v, round(out[f"{ds}|{v}"]["mean"], 2), flush=True)
        for v in ("B14", "L14", "CLIP"):
            P = data7.cub_ssb("Easy")
            sup = torch.as_tensor(P.feat(v, P.sup_ids)).cuda().reshape(P.C, 12, -1)
            mu = torch.nn.functional.normalize(sup.mean(1), dim=1)
            x = torch.as_tensor(P.feat(v, P.id_ids)).cuda()
            lab = np.array([int(c[5:]) for c in P.cls(P.id_ids)])
            out[f"cubssb|{v}"] = {"mean": 100 * float(((x @ mu.T).argmax(1).cpu().numpy() == lab).mean())}
            print("cubssb", v, round(out[f"cubssb|{v}"]["mean"], 2), flush=True)
        (RESULTS / "ncm_ds.json").write_text(json.dumps(out, indent=1) + "\n")
