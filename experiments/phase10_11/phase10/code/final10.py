"""Phase 10: the frozen Salmon Ladder V5 read-outs (static, memory p_M = M0, propagation p_LP = lp0, zero-start rank lp
and robust z) of every view on the public-benchmark streams, through the unchanged Phase 5 engine (engine5.run5, the
frozen configuration: k_g 10, gamma 1, lambda 0.9, 15 sweeps, thresholds (0.30, 0.20, 0.1019), n0 48, |K| 20, m 1,
warm start). The views of one run are independent of each other, so any product of views is formed in the analysis.
  --part openood|fourood  --views D3B,D3L | B14,L14  --worker w --nworkers n
  --check   runs the first stream of the part with the frozen views B14,L14 and compares every read-out with the
            Phase 5 final file of that stream (results_final), i.e. verifies that this pipeline reproduces Phase 5.
Output: P10/results/<part>/<views-tag>/<stream>.npz with s::<view>::<key> (log p), a::<view>::<key> (admissions),
logS (TINS), logM (MCM), is_ood, bidx, sample_id. The Phase 3 stream files fix the sample order, the batches (256) and
the TINS scores; nothing is re-drawn."""
import argparse
import json
import os
import time

import numpy as np
import torch

import p10common as P
import engine5 as E5
from static import support_dall
from vins import r5

SPEC_KW = dict(cfgs=((10, 1.0, 1.0, 0.9),), ref=(10, 1.0, 1.0, 0.9), classcond=False, neg=())


def statics(D, views, cand):
    full = {}
    for v in views:
        sup = D["shots"][v][:12000].reshape(1000, 12, -1)
        cal, ev = D["shots"][v][12000:16000], D["ev"][v]
        t0 = time.time()
        st = E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], P.V5["n0"], P.V5["m"])
        full[v] = (sup, cal, ev, st)
        print(json.dumps({"static": v, "rows": int(len(ev)), "dim": int(ev.shape[1]), "seconds": round(time.time() - t0, 1),
                          "utc": P.utc()}), flush=True)
    return full


def run_stream(D, full, pos, spec, path, device):
    run = np.load(path, allow_pickle=True)
    ids, flag = run["sample_id"], run["is_ood"].astype(bool)
    S, bidx = run["S_final"].astype(np.float64), run["batch_index"]
    rows = np.array([D["row"][x] for x in ids])
    idx = np.r_[np.arange(4000), rows]
    V = {}
    for v, (sup, cal, ev, st) in full.items():
        sts = dict(st)
        for k in ("d", "d_all", "p", "p_all", "cls", "pcos"):
            sts[k] = st[k][idx]
        V[v] = (sup, cal, ev[rows - 4000], sts)
    out, adm, timing = E5.run5(V, bidx, spec, device)
    sims = D["ev"]["CLIP"][rows - 4000] @ pos.T
    mcm = torch.softmax(torch.as_tensor(sims, dtype=torch.float64), dim=1).max(1).values.numpy()
    arrays = {"sample_id": np.array(ids), "is_ood": flag, "logS": np.log(S), "logM": np.log(mcm), "bidx": bidx}
    for v in out:
        for k, x in out[v].items():
            assert not np.isnan(x).any(), (v, k)
            arrays[f"s::{v}::{k}"] = x
        for k, x in adm[v].items():
            arrays[f"a::{v}::{k}"] = x
    return arrays


def run_part(part, views, worker, nworkers, device="cuda"):
    tag = "+".join(views)
    out_dir = P.RESULTS / part / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.npz").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    new = [v for v in views if v not in D["shots"]]
    if new:
        P.add_views(D, part, new)
    from vins import config as C
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float().numpy()
    cand = D["cands"][P.V5["K"]]
    spec = E5.make_spec(**SPEC_KW)
    P.dump(out_dir / "spec.json", {"spec_kw": {k: (list(map(list, v)) if k == "cfgs" else list(v) if isinstance(v, tuple) else v)
                                                for k, v in SPEC_KW.items()}, "views": views, "V5": {k: P.V5[k] for k in ("n0", "K", "m")},
                                   "utc": P.utc()})
    full = statics(D, views, cand)
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        arrays = run_stream(D, full, pos, spec, path, device)
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        print(json.dumps({"part": part, "views": tag, "stream": name, "n": int(len(arrays["sample_id"])),
                          "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)
        torch.cuda.empty_cache()


def check(part, device="cuda"):
    """Reproduction check of the Phase 5 final file of the first stream with the frozen views (same inputs, same engine)."""
    ds, seed, path = P.stream_tasks(part)[0]
    name = f"{ds}_seed{seed}"
    ref = np.load(P.P5 / "results_final" / part / f"{name}.npz", allow_pickle=True)
    D = P.load_p3(part)
    from vins import config as C
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float().numpy()
    cand = D["cands"][P.V5["K"]]
    full = statics(D, P.BASE_VIEWS, cand)
    arrays = run_stream(D, full, pos, E5.make_spec(**SPEC_KW), path, device)
    assert np.array_equal(arrays["sample_id"], ref["sample_id"])
    rep = {}
    for k in arrays:
        if k.startswith("s::") and k in ref.files:
            rep[k] = float(np.max(np.abs(arrays[k] - ref[k])))
    rep["logS"] = float(np.max(np.abs(arrays["logS"] - ref["logS"])))
    rep["logM"] = float(np.max(np.abs(arrays["logM"] - ref["logM"])))
    worst = max(rep.values())
    P.dump(P.RESULTS / f"check_{part}.json", {"stream": name, "max_abs_diff": rep, "worst": worst, "utc": P.utc()})
    print(json.dumps({"check": part, "stream": name, "worst_abs_diff": worst,
                      "keys": {k: round(v, 6) for k, v in rep.items()}}), flush=True)
    print("CHECK_OK" if worst < 1e-6 else "CHECK_DIFFERS", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--views", default="D3B,D3L")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.check:
        check(a.part)
    else:
        run_part(a.part, a.views.split(","), a.worker, a.nworkers)
