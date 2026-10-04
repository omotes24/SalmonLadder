"""Phase 5 final evaluation of the locked method (run once, after the lock):
  --part u4       the unused bank U4 (5 splits x 3 draws x 3 orders)
  --part openood  OpenOOD v1.5 ImageNet-1K test (5 OOD sets x 5 orders; 6th use of the test split overall)
  --part fourood  Four-OOD (4 OOD sets x 3 orders)
The frozen REPRISE v5 read-outs come out of the same run (unchanged Phase 4 code paths), so every comparison is paired.
Output: results_final/<part>/<stream>.npz (log p of every read-out per view, TINS and MCM scores, labels for the
evaluation scripts only)."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import GONOGO, V5, utc
import engine5 as E5
from static import support_dall
from u4_common import P5, require_lock
from vins import r5

OUT = P5 / "results_final"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]


def lock_cfg(L):
    spec = dict(L["spec"])
    for k in ("cfgs", "negcfgs"):
        if spec.get(k) is not None:
            spec[k] = [tuple(c) for c in spec[k]]
    if "ref" in spec:
        spec["ref"] = tuple(spec["ref"])
    return [tuple(v) for v in L["views"]], E5.make_spec(**spec)


def save(part, name, arrays, out, adm, t0, timing):
    for v in out:
        for k, x in out[v].items():
            assert not np.isnan(x).any(), (v, k)
            arrays[f"s::{v}::{k}"] = x
        for k, x in adm[v].items():
            arrays[f"a::{v}::{k}"] = x
    d = OUT / part
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, d / f"{name}.npz")
    print(json.dumps({"part": part, "stream": name, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


def transform(views_cfg, raw, device):
    """raw: {view: (sup, cal, sf)}; applies the locked feature map of every view (fitted on its support)."""
    out = {}
    for (vname, kind, alpha) in views_cfg:
        sup, cal, sf = raw[vname]
        if kind != "raw":
            f = E5.fit_transform(sup, kind, alpha, device)
            sup, cal, sf = f(sup), f(cal), f(sf)
        out[vname] = (sup, cal, sf)
    return out


def run_u4(L, worker, nworkers, device):
    from u4_bank import Features, stream_inputs
    views_cfg, spec = lock_cfg(L)
    names = sorted({n for v in views_cfg for n in v[0].split("+")})
    F = Features(tuple(sorted(set(names) | {"CLIP"})))
    tasks = [(s, d, sd) for s in range(1, 6) for d in range(3) for sd in (123, 124, 125)][worker::nworkers]
    for (s, d, sd) in tasks:
        name = f"U4_s{s}_d{d}_seed{sd}"
        if (OUT / "u4" / f"{name}.npz").exists():
            continue
        t0 = time.time()
        inp = stream_inputs(F, s, d, sd, views=names)
        raw = {}
        for (vname, kind, alpha) in views_cfg:
            parts = vname.split("+")
            trip = [(inp[f"sup_{p}"], inp[f"cal_{p}"], inp[f"sf_{p}"]) for p in parts]
            raw[vname] = trip[0] if len(parts) == 1 else tuple(E5.l2n(np.concatenate([E5.l2n(t[i]) for t in trip], -1)) for i in range(3))
        V = {}
        for vname, (sup, cal, sf) in transform(views_cfg, raw, device).items():
            V[vname] = (sup, cal, sf, E5.static5(r5.proto_view, support_dall, sup, cal, sf, inp["cand_c"], inp["cand_s"], V5["n0"], V5["m"]))
        out, adm, timing = E5.run5(V, inp["bidx"], spec, device)
        save("u4", name, {"sample_id": np.array(inp["ids"]), "is_ood": inp["flag"], "logS": np.log(inp["S"]),
                          "logM": np.log(inp["mcm"]), "bidx": inp["bidx"], "wnid": inp["wnid"]}, out, adm, t0, timing)


def run_test(L, part, worker, nworkers, device):
    spec_ = importlib.util.spec_from_file_location("p3_eval", GONOGO / "scripts" / "p3_eval.py")
    P3E = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(P3E)
    from vins import config as C
    views_cfg, spec = lock_cfg(L)
    D = P3E.load_part(part)
    if part == "openood":
        tasks = [(ds, s, C.WORK / "test_runs" / "default" / f"{ds}_seed{s}.npz") for ds in OO for s in (123, 124, 125, 126, 127)]
    else:
        tasks = [(o, s, C.WORK / "extra_runs" / "fourood" / f"imagenet_{o}_seed{s}.npz") for o in FOUR for s in (123, 124, 125)]
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float().numpy()
    # static view once per view over [calibration; every evaluated image] (ranks use the calibration images only,
    # so the rows of a stream can be sliced out afterwards, as Phase 3 did)
    cand = D["cands"][V5["K"]]
    full = {}
    for (vname, kind, alpha) in views_cfg:
        parts = vname.split("+")
        trip = [(D["shots"][p][:12000].reshape(1000, 12, -1), D["shots"][p][12000:16000], D["ev"][p]) for p in parts]
        raw = trip[0] if len(parts) == 1 else tuple(E5.l2n(np.concatenate([E5.l2n(t[i]) for t in trip], -1)) for i in range(3))
        sup, cal, ev = transform([(vname, kind, alpha)], {vname: raw}, device)[vname]
        full[vname] = (sup, cal, ev, E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], V5["n0"], V5["m"]))
        print(json.dumps({"static": vname, "rows": int(len(ev)), "utc": utc()}), flush=True)
    for (ds, seed, path) in tasks[worker::nworkers]:
        name = f"{ds}_seed{seed}"
        if (OUT / part / f"{name}.npz").exists():
            continue
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        S, bidx = run["S_final"].astype(np.float64), run["batch_index"]
        rows = np.array([D["row"][x] for x in ids])
        idx = np.r_[np.arange(4000), rows]
        V = {}
        for vname, (sup, cal, ev, st) in full.items():
            sts = dict(st)
            for k in ("d", "d_all", "p", "p_all", "cls", "pcos"):
                sts[k] = st[k][idx]
            V[vname] = (sup, cal, ev[rows - 4000], sts)
        out, adm, timing = E5.run5(V, bidx, spec, device)
        sims = D["ev"]["CLIP"][rows - 4000] @ pos.T
        mcm = torch.softmax(torch.as_tensor(sims, dtype=torch.float64), dim=1).max(1).values.numpy()
        save(part, name, {"sample_id": np.array(ids), "is_ood": flag, "logS": np.log(S), "logM": np.log(mcm), "bidx": bidx}, out, adm, t0, timing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["u4", "openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    L = require_lock()
    if a.part == "u4":
        run_u4(L, a.worker, a.nworkers, "cuda")
    else:
        run_test(L, a.part, a.worker, a.nworkers, "cuda")
