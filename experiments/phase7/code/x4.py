"""Phase 7 intervention (the registered Exp 4 design of Phase 4, same-class content only) with every engine7 read-out.
For each probe class c and r in {0,1,2,5,10,20}: the fixed history (768 ID + 256 OOD) with r nested OOD positions
replaced by images of c is streamed (batches of 256), then c's 8 queries and the 64 common ID queries are each scored
alone on a clone of the state. Scores are sums over the two views (log p).
Output: <P7>/results/x4/s<k>_r0.parquet, s<k>_<wnid>.parquet (r, query, key, score)."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import data7
import engine7 as E7
from p7common import P4, RESULTS, V5, utc
from static import static_view

OUT = RESULTS / "x4"
VIEWS = ("B14", "L14")


def run_split(k, classes):
    dz = json.loads((P4 / "banks" / "exp4" / f"split{k}.json").read_text())
    P = data7.u1(k, 0)
    C = P.C
    probes = dz["probes"]
    need = list(dict.fromkeys(dz["history"] + dz["id_queries"] + [q for w in dz["probe"] for q in probes[w]["queries"]]
                              + [s for w in dz["probe"] for s in probes[w]["same"]]))
    pos = {s: i for i, s in enumerate(need)}
    clip_need, clip_cal = P.feat("CLIP", need), P.feat("CLIP", P.cal_ids)
    cand_n = np.argsort(-(clip_need @ P.pos.T), axis=1)[:, :V5["K"]]
    cand_c = np.argsort(-(clip_cal @ P.pos.T), axis=1)[:, :V5["K"]]
    V = {}
    for v in VIEWS:
        sup = P.feat(v, P.sup_ids).reshape(C, 12, -1)
        cal = P.feat(v, P.cal_ids)
        X = P.feat(v, need)
        st = static_view(sup, cal, X, cand_c, cand_n, V5["n0"], V5["m"])
        nc = len(cal)
        V[v] = {"sup": sup, "cal": cal, "X": X, "st": st, "d": st["d"][nc:], "da": st["d_all"][nc:], "p": st["p"][nc:]}

    def stream_state(history):
        ix = np.array([pos[s] for s in history])
        states = {}
        for v in VIEWS:
            S = E7.State7(V[v]["sup"], V[v]["cal"], V[v]["st"], V5["thresholds"], k=V5["kg"], gamma=V5["gamma"], lam=V5["lam"])
            for lo in range(0, len(ix), 256):
                r = ix[lo:lo + 256]
                S.step(V[v]["X"][r], V[v]["d"][r], V[v]["da"][r])
            states[v] = S
        return states

    def probe_all(states, qids):
        rows = []
        for q in qids:
            i = pos[q]
            sc = {}
            for v in VIEWS:
                o = states[v].probe(V[v]["X"][i], V[v]["d"][i], V[v]["da"][i], V[v]["p"][i])
                for key, val in o.items():
                    if key == "rho":
                        sc[f"rho_{v}"] = val
                    else:
                        sc[key] = sc.get(key, 0.0) + val
            rows.append((q, sc))
        return rows

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"s{k}_r0.parquet"
    if not base.exists():
        all_q = dz["id_queries"] + [q for w in dz["probe"] for q in probes[w]["queries"]]
        rows = [{"r": 0, "query": q, "key": key, "score": s} for q, sc in probe_all(stream_state(dz["history"]), all_q) for key, s in sc.items()]
        tmp = OUT / f"s{k}_r0.{os.getpid()}.parquet"
        pd.DataFrame(rows).assign(split=k).to_parquet(tmp, index=False)
        os.replace(tmp, base)
    for w in classes:
        f_out = OUT / f"s{k}_{w}.parquet"
        if f_out.exists():
            continue
        t0 = time.time()
        pr = probes[w]
        rows = []
        for r in (1, 2, 5, 10, 20):
            hist = list(dz["history"])
            for j, p_ in enumerate(dz["replace_order"][:r]):
                hist[p_] = pr["same"][j]
            for q, sc in probe_all(stream_state(hist), dz["id_queries"] + pr["queries"]):
                for key, s in sc.items():
                    rows.append({"r": r, "query": q, "key": key, "score": s})
        tmp = OUT / f"s{k}_{w}.{os.getpid()}.parquet"
        pd.DataFrame(rows).assign(split=k, wnid=w).to_parquet(tmp, index=False)
        os.replace(tmp, f_out)
        print(json.dumps({"x4": f"s{k}_{w}", "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    tasks = []
    for k in range(1, 6):
        dz = json.loads((P4 / "banks" / "exp4" / f"split{k}.json").read_text())
        tasks += [(k, w) for w in dz["probe"]]
    mine = tasks[a.worker::a.nworkers]
    if a.limit:
        mine = mine[:a.limit]
    for k in range(1, 6):
        cls = [w for kk, w in mine if kk == k]
        if cls:
            run_split(k, cls)
