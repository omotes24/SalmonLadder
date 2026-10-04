"""Exp 4: recurrence intervention on U1 held-out probe classes (draw 0, s0 = 1, batches of 256 for the history).
For each probe class c, content X in {same, near, far, dup} and r in {0,1,2,5,10,20}: the fixed history with r
nested OOD positions replaced is streamed, then c's 8 queries and 64 common ID queries are each scored alone on a
cloned state. Families: static, memory p_t, p_LP (REPRISE LP), frozen REPRISE, static x p_LP, best minimal."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bank_data import Features, pos_text, shots_of, spec
from common import BANKS, FEATS, RESULTS, V5, utc
from evaluate import load_lock, parse
from state import StreamState
from static import static_view

OUT = RESULTS / "exp4"
REF = ("cdf_L0", 10, 1.0, 0.9)


class AugFeatures:
    def __init__(self):
        t = pd.read_parquet(BANKS / "exp4" / "aug_table.parquet")
        self.row = dict(zip(t.sample_id, range(len(t))))
        self.F = {n: np.load(FEATS / f"aug_{n}.npy", mmap_mode="r") for n in ("B14", "L14", "CLIP")}


def feats(F, A, name, ids):
    out = np.empty((len(ids), F.F[name].shape[1]), np.float32)
    a = [i for i, s in enumerate(ids) if s.startswith("aug/")]
    b = [i for i, s in enumerate(ids) if not s.startswith("aug/")]
    if b:
        out[b] = F.get(name, [ids[i] for i in b])
    if a:
        out[a] = np.asarray(A.F[name][[A.row[ids[i]] for i in a]], np.float32)
    return out


def best_cfg(lock):
    f = lock["best_minimal"]
    k, g, l = parse(lock["selected"][f]["config"])
    base = {"stat_cdf": "cdf", "stat_cdf_L0": "cdf_L0"}.get(f, f)
    return f, (base, k, g, l)


def run_split(k, classes, F, A, lock):
    best, bcfg = best_cfg(lock)
    cfgs = [REF] + ([bcfg] if bcfg != REF else [])
    dz = json.loads((BANKS / "exp4" / f"split{k}.json").read_text())
    names, shots, _, _ = spec("U1", k)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, 0, C)
    probes = dz["probes"]
    need = list(dict.fromkeys(dz["history"] + dz["id_queries"] + [q for w in dz["probe"] for q in probes[w]["queries"]]
                              + [s for w in dz["probe"] for s in probes[w]["same"]]
                              + [f"aug/s{k}/{w}/{j}" for w in dz["probe"] for j in range(20)]))
    pos = {s: i for i, s in enumerate(need)}
    pt = pos_text("U1", k)
    clip_need, clip_cal = feats(F, A, "CLIP", need), F.get("CLIP", cal_ids)
    cand_n = np.argsort(-(clip_need @ pt.T), axis=1)[:, :20]
    cand_c = np.argsort(-(clip_cal @ pt.T), axis=1)[:, :20]
    V = {}
    for v in ("B14", "L14"):
        sup = F.get(v, sup_ids).reshape(C, 12, -1)
        cal = F.get(v, cal_ids)
        X = feats(F, A, v, need)
        st = static_view(sup, cal, X, cand_c, cand_n, V5["n0"], V5["m"])
        nc = len(cal)
        V[v] = {"sup": sup, "cal": cal, "X": X, "st": st, "d": st["d"][nc:], "da": st["d_all"][nc:], "p": st["p"][nc:]}

    def stream_state(history):
        states = {}
        ix = np.array([pos[s] for s in history])
        for v in ("B14", "L14"):
            S = StreamState(V[v]["sup"], V[v]["cal"], V[v]["st"], V5["thresholds"], cfgs)
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
            for v in ("B14", "L14"):
                o = states[v].probe(V[v]["X"][i], V[v]["d"][i], V[v]["da"][i], V[v]["p"][i])
                for key, val in o.items():
                    sc[key] = sc.get(key, 0.0) + val
            fam = {"static": sc["static"], "Mpt": sc["Mpt"], "pLP": sc[REF], "REPRISE": sc["Mpt"] + sc[REF],
                   "static_pLP": sc["static"] + sc[REF]}
            b = sc[bcfg]
            fam["best"] = b + (sc["static"] if best.startswith("stat_") else 0.0)
            rows.append((q, fam))
        return rows

    OUT.mkdir(parents=True, exist_ok=True)
    base_file = OUT / f"s{k}_r0.parquet"
    all_q = dz["id_queries"] + [q for w in dz["probe"] for q in probes[w]["queries"]]
    if not base_file.exists():
        st0 = stream_state(dz["history"])
        rows = [{"query": q, "family": f, "score": s} for q, fam in probe_all(st0, all_q) for f, s in fam.items()]
        tmp = OUT / f"s{k}_r0.{os.getpid()}.parquet"
        pd.DataFrame(rows).assign(split=k).to_parquet(tmp, index=False)
        os.replace(tmp, base_file)
    for w in classes:
        f_out = OUT / f"s{k}_{w}.parquet"
        if f_out.exists():
            continue
        t0 = time.time()
        pr = probes[w]
        content = {"same": pr["same"], "near": probes[pr["near"]]["same"], "far": probes[pr["far"]]["same"],
                   "dup": [f"aug/s{k}/{w}/{j}" for j in range(20)]}
        rows = []
        for cond, src in content.items():
            for r in (1, 2, 5, 10, 20):
                hist = list(dz["history"])
                for j, p_ in enumerate(dz["replace_order"][:r]):
                    hist[p_] = src[j]
                st = stream_state(hist)
                for q, fam in probe_all(st, dz["id_queries"] + pr["queries"]):
                    for f, s in fam.items():
                        rows.append({"cond": cond, "r": r, "query": q, "family": f, "score": s})
        tmp = OUT / f"s{k}_{w}.{os.getpid()}.parquet"
        pd.DataFrame(rows).assign(split=k, wnid=w, best_family=best).to_parquet(tmp, index=False)
        os.replace(tmp, f_out)
        print(json.dumps({"exp4": f"s{k}_{w}", "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    lock = load_lock()
    F, A = Features(), AugFeatures()
    tasks = []
    for k in range(1, 6):
        dz = json.loads((BANKS / "exp4" / f"split{k}.json").read_text())
        tasks += [(k, w) for w in dz["probe"]]
    mine = tasks[a.worker::a.nworkers]
    for k in range(1, 6):
        cls = [w for kk, w in mine if kk == k]
        if cls:
            run_split(k, cls, F, A, lock)
