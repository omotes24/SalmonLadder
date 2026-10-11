"""Phase 16 smoke test (no dev2, no test-set scoring):
(1) the calibration images are stored class-major (blocks of 4 per class) on dev1, dev2 (labels only), the OpenOOD shots of the
    paper's draw and the pool draws (paths / tables only) -> R = positions 0, 1 and C = positions 2, 3 are 2 per class;
(2) core16.h16 on dev1 draw 0 (far, near) reproduces the exploration value of harm16d.py ('pool2_unionN_l0.95_gr0.0625':
    FPR95 17.583664 far, 13.74 near; seeds 1471 / 5049) — dev1 has been explored already;
(3) the conjugate-gradient solver against a dense solve on a small random graph.
Writes P16/results/smoke16.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p16common as P
import core14 as K
import core16 as H
from an11 import measures

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402

VIEWS = list(P.VIEWS16)
EXPECT = {"far": (17.583664, 1471), "near": (13.74, 5049)}


def blocks_ok(labels, per=4):
    labels = np.asarray(labels)
    if len(labels) % per:
        return False
    b = labels.reshape(-1, per)
    return bool((b == b[:, :1]).all() and len(set(b[:, 0].tolist())) == len(b))


def main():
    res = {"utc": P.utc()}
    # (1) class-major calibration order (the OpenOOD paths first: loading a dev split re-points the vins work directory)
    paths = P.image_paths("openood")
    res["cal_blocks_openood_p3"] = blocks_ok([Path(p).parts[-2] for p in paths[12000:16000]])
    res["sup_blocks_openood_p3"] = blocks_ok([Path(p).parts[-2] for p in paths[:12000]], per=12)
    pool = pd.read_parquet(P.P.P4 / "banks" / "imagenet_pool.parquet")
    pb = {}
    for d in range(5):
        s = pool[(pool.role == "shot") & (pool.draw == d)].sort_values(["idx_1k", "pos"])
        pb[str(d)] = blocks_ok(s[s.pos >= 12].idx_1k.to_numpy())
    res["cal_blocks_pool"] = pb
    for dev in ("dev1", "dev2"):
        D = dev5.load(dev)                                   # shot ids are paths 'in1k/train/<wnid>/<file>'
        res[f"cal_blocks_{dev}"] = {str(d): blocks_ok([Path(s).parts[-2] for s in D["draws"][str(d)]["cal"]]) for d in sorted(D["draws"])}
        if dev == "dev1":
            D1 = D
    # (3) solver check on a small graph
    rng = np.random.default_rng(0)
    n = 300
    A = rng.random((n, n)) * (rng.random((n, n)) < 0.05)
    Wd = np.maximum(A, A.T)
    np.fill_diagonal(Wd, 0.0)
    d = Wd.sum(1)
    one = np.zeros(n, bool)
    one[:20] = True
    zero = np.zeros(n, bool)
    zero[20:30] = True
    F = ~(one | zero)
    lam, gam = 0.95, 0.3
    M = np.diag(d[F] + gam) - lam * Wd[np.ix_(F, F)]
    hd = np.zeros(n)
    hd[one] = 1.0
    hd[F] = np.linalg.solve(M, lam * Wd[np.ix_(F, one)].sum(1))
    Wt = torch.as_tensor(Wd, dtype=torch.float64, device="cuda").to_sparse_csr()
    hc, it, rr = H.solve_h(Wt, torch.as_tensor(d, device="cuda"), torch.as_tensor(one, device="cuda"), torch.as_tensor(zero, device="cuda"), lam, gam)
    res["solver_max_abs_err"] = float(np.abs(hc.cpu().numpy() - hd).max())
    # (2) reproduction on dev1 draw 0
    rep = {}
    for stream in ("far", "near"):
        t0 = time.time()
        key = sorted(k for k in D1["tins"] if k[0] == "0" and k[1] == stream)[0]
        sid, flag, S, _ = D1["tins"][key]
        flag = np.asarray(flag, bool)
        rows = np.array([D1["stream_row"][s] for s in sid])
        graphs = {}
        for v in VIEWS:
            sup, cal, sf = dev5.raw_view(D1, "0", v, rows)
            graphs[v] = K.Graph(P.l2n(sup.reshape(-1, sup.shape[-1])), P.l2n(cal), P.l2n(sf), "cuda")
        out = H.h16(graphs)
        au, fp = measures(H.logc(out["s"]), flag)
        rep[stream] = {"FPR95": fp, "AUROC": au, "seeds": out["info"]["seeds"], "expected": EXPECT[stream],
                       "match": bool(abs(fp - EXPECT[stream][0]) < 0.05 and out["info"]["seeds"] == EXPECT[stream][1]),
                       "validity": H.validity(out["p"], flag), "info": out["info"], "seconds": round(time.time() - t0, 1)}
        for g in graphs.values():
            g.free()
    res["dev1_reproduction"] = rep
    res["ok"] = bool(all(res[f"cal_blocks_{d}"][k] for d in ("dev1", "dev2") for k in res[f"cal_blocks_{d}"]) and res["cal_blocks_openood_p3"]
                     and all(pb.values()) and res["solver_max_abs_err"] < 1e-8 and all(r["match"] for r in rep.values()))
    P.write_json(P.RESULTS16 / "smoke16.json", res)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE16_OK" if res["ok"] else "SMOKE16_FAIL", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
