"""Phase 7, experiment A: every engine7 read-out on the development streams (dev1: selection, dev2: confirmation).
Only the dev caches are read. Output: <P7>/results/dev/<view>/<dev>_draw<d>_<stream>_seed<s>.npz"""
import argparse
import json
import time

import numpy as np
import torch

import p7common  # noqa: F401  (paths)
import dev5
import engine5 as E5
import engine7 as E7
from p7common import RESULTS, V5, utc
from run7 import save
from static import support_dall
from vins import r5


def run_task(D, draw, stream, seed, view, retro=()):
    name = f"{D['dev']}_draw{draw}_{stream}_seed{seed}"
    exp = "dev" if not retro else ("devw" if tuple(retro) == (0,) else "devr")
    if (RESULTS / exp / view / f"{name}.npz").exists():
        return
    t0 = time.time()
    sid, flag, S, bidx = D["tins"][(str(draw), stream, seed)]
    rows = np.array([D["stream_row"][s] for s in sid])
    dr = D["draws"][str(draw)]
    cand_cal = np.argsort(-dr["cal_sims"], axis=1)[:, :V5["K"]]
    cand_sf = D["cand_stream"][rows]
    sup, cal, sf = dev5.raw_view(D, draw, view, rows)
    st = E5.static5(r5.proto_view, support_dall, sup, cal, sf, cand_cal, cand_sf, V5["n0"], V5["m"])
    out, aux = E7.run_view7(sup, cal, sf, bidx, st, V5["thresholds"], k=V5["kg"], gamma=V5["gamma"], lam=V5["lam"], retro=retro)
    arr = {"sample_id": np.array(sid), "is_ood": np.asarray(flag, bool), "cls": D["wnid"][rows].astype(str), "bidx": np.asarray(bidx),
           "logM": np.log(D["mcm"][rows]), "logS": np.log(S.astype(np.float64)), "admit": aux["admit"], "rho": aux["rho"],
           "d": st["d"][len(cal):].astype(np.float32)}
    for k, v in out.items():
        arr[f"s::{k}"] = v
    arr["meta"] = np.array(json.dumps({"problem": name, "view": view, "seconds": aux["seconds"]}))
    save(exp, view, name, arr)
    torch.cuda.empty_cache()
    print(json.dumps({"exp": exp, "view": view, "task": name, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="dev1")
    ap.add_argument("--views", default="B14,L14")
    ap.add_argument("--draws", default="0,1,2,3,4")
    ap.add_argument("--streams", default="near,far")
    ap.add_argument("--seeds", default="123,124,125")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retro", action="store_true", help="delayed re-scoring read-outs -> results/devr")
    ap.add_argument("--within", action="store_true", help="within-batch memory read-out ('@0') -> results/devw")
    a = ap.parse_args()
    torch.set_num_threads(2)
    D = dev5.load(a.dev)
    tasks = [(int(d), s, int(sd), v) for d in a.draws.split(",") for s in a.streams.split(",") for sd in a.seeds.split(",")
             for v in a.views.split(",")][a.worker::a.nworkers]
    from run7 import RETRO
    for t in tasks[:a.limit or None]:
        run_task(D, *t, retro=(0,) if a.within else (RETRO if a.retro else ()))
