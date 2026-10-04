"""Phase 5 exploration on the development splits (dev1: selection, dev2: confirmation).

Only the dev caches are read (features, TINS scores, shots of the five R5 draws). No unused bank, no OpenOOD test
image and no ImageNet val image is opened here. The engine never receives is_ood; labels and class names are saved
next to the scores for the evaluation scripts only.

Output: results/p5/<runcfg>/<dev>_draw<d>_<stream>_seed<s>.npz  (log p of every read-out, per view)
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from common import AUDIT, ROOT, V5, utc
import engine5 as E5
from static import support_dall
from vins import r5

OUT = Path("/home/omote/reprise_p5_20261002/results")
REF = (10, 1.0, 1.0, 0.9)
CFGS_FULL = [REF, (5, 1.0, 1.0, 0.9), (7, 1.0, 1.0, 0.9), (20, 1.0, 1.0, 0.9), (10, 0.0, 1.0, 0.9), (10, 3.0, 1.0, 0.9),
             (10, 1.0, 0.5, 0.9), (10, 1.0, 0.0, 0.9), (10, 1.0, 1.0, 0.8), (20, 1.0, 1.0, 0.8)]
CFGS_SMALL = [REF, (10, 0.0, 1.0, 0.9), (20, 1.0, 1.0, 0.9), (5, 1.0, 1.0, 0.9)]
DINO = [("B14", "raw", 0.0), ("L14", "raw", 0.0)]
RUN = {
    "base": {"views": DINO, "spec": dict(cfgs=CFGS_FULL, eps_lp=(0.05, 0.1), eps_fin=(0.1,), eps_joint=(0.05, 0.1, 0.2, 0.4),
                                         neg=("M0", "Mjoint0.1"))},
    "center": {"views": [("B14", "center", 0.0), ("L14", "center", 0.0)],
               "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "white03": {"views": [("B14", "white", 0.3), ("L14", "white", 0.3)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "white06": {"views": [("B14", "white", 0.6), ("L14", "white", 0.6)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "white09": {"views": [("B14", "white", 0.9), ("L14", "white", 0.9)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "clip3": {"views": DINO + [("CLIP", "raw", 0.0)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1, 0.2), neg=("M0",))},
    "clip3c": {"views": DINO + [("CLIP", "center", 0.0)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "fuse": {"views": [("B14+L14", "raw", 0.0)], "spec": dict(cfgs=CFGS_SMALL, eps_fin=(0.1,), eps_joint=(0.1,), neg=("M0",))},
    "fuse3": {"views": DINO + [("B14+L14", "raw", 0.0)], "spec": dict(cfgs=CFGS_SMALL, eps_joint=(0.1,), neg=("M0",))},
    "fuseall": {"views": [("B14+L14+CLIP", "raw", 0.0)], "spec": dict(cfgs=CFGS_SMALL, eps_fin=(0.1,), neg=("M0",))},
    "tier2": {"views": DINO, "spec": dict(cfgs=[REF], eps_joint=(0.1,), neg=("M0",), tier2=True)},
    # round 2: two-sided propagation (OOD seeds), full-score gate, centroid read-outs, oracle seeds (diagnostic)
    "r2": {"views": DINO, "spec": dict(
        cfgs=[REF, (10, 0.0, 1.0, 0.9)], eps_joint=(0.05, 0.1, 0.2, 0.3), eps_full=(0.05, 0.1, 0.2, 0.3), frz=True, oracle=True,
        neg=("M0", "Mjoint0.05", "Mjoint0.1", "Mjoint0.2", "Mjoint0.3", "Mora"),
        negcfgs=[REF, (10, 1.0, 1.0, 0.8), (10, 1.0, 1.0, 0.5), (10, 0.0, 1.0, 0.9), (20, 1.0, 1.0, 0.9), (5, 1.0, 1.0, 0.9)],
        cen=("Mfrz", "Mjoint0.1", "Mjoint0.2", "Mfull0.1", "Mfull0.2", "Mora"), cen_k=10, classcond=False)},
    # round 3: dynamic seed sets (current propagation evidence of earlier images) with neighbourhood refinement
    "r3": {"views": DINO, "spec": dict(
        cfgs=[REF], eps_joint=(0.1,), neg=("M0", "Mjoint0.1"), classcond=False,
        dyn=[(s, e, m, False) for s in ("lp", "lpst") for e in (0.05, 0.1, 0.2) for m in (0, 3, 5)] +
            [(s, e, 3, True) for s in ("lp", "lpst") for e in (0.05, 0.1, 0.2)])},
    # round 4: seed quality (hysteresis, soft weights, per-view rules, a second round with the received seed mass)
    "r4": {"views": DINO, "spec": dict(
        cfgs=[REF], neg=("M0",), classcond=False,
        dyn=[("lp", e, 0, False) for e in (0.02, 0.05, 0.1, 0.2)] +
            [dict(kind="hyst", lo=lo, hi=hi, m=m) for lo, hi, m in ((0.05, 0.2, 2), (0.05, 0.3, 2), (0.05, 0.3, 3), (0.02, 0.2, 2),
                                                                     (0.02, 0.1, 1), (0.1, 0.3, 3), (0.05, 0.5, 4))] +
            [dict(kind="soft", eps=e) for e in (0.05, 0.1, 0.2, 0.5, 1.0)] +
            [dict(kind=k, eps=e) for k, e in (("own", 0.05), ("own", 0.1), ("any", 0.05), ("any", 0.1), ("all", 0.1), ("all", 0.2))] +
            [dict(kind="iter", base=b, eps=e) for b, e in (("Dlp0.05m0", 0.05), ("Dlp0.05m0", 0.1), ("Dlp0.1m0", 0.05),
                                                           ("Dlp0.1m0", 0.1), ("Dlp0.2m0", 0.1))])},
    # round 5: seeds = Benjamini-Hochberg rejections among the earlier images (FDR-controlled self-labelling)
    "r5": {"views": DINO, "spec": dict(
        cfgs=[REF], neg=("M0",), classcond=False,
        dyn=[("lp", 0.05, 0, False), ("lp", 0.1, 0, False)] +
            [dict(kind="bh", q=q) for q in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)] +
            [dict(kind="bh", q=q, storey=True) for q in (0.05, 0.1, 0.2, 0.3)] +
            [dict(kind="iter", base=b, q=q) for b, q in (("B0.1", 0.1), ("B0.2", 0.2), ("B0.2", 0.1), ("B0.3", 0.2), ("B0.1s", 0.1), ("B0.2s", 0.2))] +
            [dict(kind="iter", base=b, q=q, storey=True) for b, q in (("B0.1s", 0.1), ("B0.2s", 0.2))])},
    # round 6: finer BH levels and partial re-calibration of factor groups
    "r6": {"views": DINO, "spec": dict(
        cfgs=[REF], neg=("M0",), classcond=False, recal=True,
        dyn=[dict(kind="bh", q=q, storey=True) for q in (0.2, 0.25, 0.3, 0.35, 0.4)] + [dict(kind="bh", q=q) for q in (0.3, 0.4)] +
            [dict(kind="iter", base=b, q=q, storey=True) for b, q in (("B0.2s", 0.2), ("B0.3s", 0.3), ("B0.3s", 0.2))])},
    # round 7: two and three rounds of Storey-BH self-labelling
    "r7": {"views": DINO, "spec": dict(
        cfgs=[REF], neg=("M0",), classcond=False,
        dyn=[dict(kind="bh", q=q, storey=True) for q in (0.3, 0.4, 0.5)] +
            [dict(kind="iter", base=f"B{q1:g}s", q=q2, storey=True) for q1 in (0.3, 0.4, 0.5) for q2 in (0.1, 0.15, 0.2, 0.25)] +
            [dict(kind="iter", base=b, q=q, storey=True) for b, q in (("IB0.2s<B0.3s", 0.2), ("IB0.2s<B0.3s", 0.15), ("IB0.2s<B0.4s", 0.2),
                                                                      ("IB0.25s<B0.5s", 0.2), ("IB0.25s<B0.3s", 0.2))])},
}


LOCK = Path("/home/omote/reprise_p5_20261002/selection_lock_p5.json")


def locked_cfg():
    """Run configuration of the locked method (selection_lock_p5.json; sha256 checked)."""
    from common import sha_file
    want = LOCK.with_suffix(".sha256").read_text().split()[0]
    assert sha_file(LOCK) == want, "selection lock changed"
    L = json.loads(LOCK.read_text())
    spec = dict(L["spec"])
    for k in ("cfgs", "negcfgs"):
        if spec.get(k) is not None:
            spec[k] = [tuple(c) for c in spec[k]]
    if "ref" in spec:
        spec["ref"] = tuple(spec["ref"])
    return {"views": [tuple(v) for v in L["views"]], "spec": spec}


def load(dev):
    spec = importlib.util.spec_from_file_location("round2", AUDIT / "vendor" / "scripts" / "r5_round2.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    D = R.load_dev(dev)
    from vins import config as C
    from vins.features import load_features

    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    stream_ids = [None] * len(D["stream_row"])
    for s, i in D["stream_row"].items():
        stream_ids[i] = s
    wn = dict(zip(samples.sample_id, samples.wnid.fillna("").astype(str))) if "wnid" in samples else {}
    D["wnid"] = np.array([wn.get(s, "") for s in stream_ids])
    clip_dev, cb = load_features(C.FEATURES_DIR / "clip.pt")
    cpos = {s: i for i, s in enumerate(cb["sample_id"])}
    shot_clip, sb = load_features(r5.R5 / "features" / "shots.clip.pt")
    D["feats"]["CLIP"] = {"stream": clip_dev[[cpos[s] for s in stream_ids]].float().numpy().astype(np.float32),
                          "shots": shot_clip.float().numpy(), "shot_pos": {s: i for i, s in enumerate(sb["sample_id"])}}
    extra = Path("/home/omote/reprise_p5_20261002/features")
    for name in ("B14pm", "L14pm", "G14", "G14pm", "SIG2L", "CLIPL",           # second-stage views (if extracted)
                 "D3B", "D3L", "D3Bs", "D3Ls"):                                # Phase 6: DINOv3 views (extract6.py)
        fs, fd = extra / f"shots.{name}.pt", extra / f"{dev}.{name}.pt"
        if fs.exists() and fd.exists():
            s, d_ = torch.load(fs, map_location="cpu"), torch.load(fd, map_location="cpu")
            pos = {x: i for i, x in enumerate(d_["sample_id"])}
            D["feats"][name] = {"stream": d_["features"][[pos[x] for x in stream_ids]].float().numpy().astype(np.float32),
                                "shots": s["features"].float().numpy(), "shot_pos": {x: i for i, x in enumerate(s["sample_id"])}}
    D["cand_stream"] = np.argsort(-D["st_sims"], axis=1)[:, :V5["K"]]
    D["mcm"] = torch.softmax(torch.as_tensor(D["st_sims"], dtype=torch.float64), dim=1).max(1).values.numpy()
    return D


def raw_view(D, draw, name, rows):
    """Support (C x 12 x D), calibration and stream features of one view, as the frozen pipeline reads them."""
    if "+" in name:
        parts = [raw_view(D, draw, n, rows) for n in name.split("+")]
        return tuple(E5.l2n(np.concatenate([E5.l2n(p[i]) for p in parts], axis=-1)) for i in range(3))
    F, dr = D["feats"][name], D["draws"][str(draw)]
    sup = F["shots"][[F["shot_pos"][s] for s in dr["sup"]]].astype(np.float32).reshape(D["n_id"], 12, -1)
    cal = F["shots"][[F["shot_pos"][s] for s in dr["cal"]]].astype(np.float32)
    return sup, cal, F["stream"][rows]


# Phase 6 (encoder swap, no tuning): the locked specification with the two DINOv2 views replaced by DINOv3 views
SWAP = {"d3": [("D3B", "raw", 0.0), ("D3L", "raw", 0.0)],        # ViT-B/16 + ViT-L/16 at 256 px (pre-registered primary)
        "d3s": [("D3Bs", "raw", 0.0), ("D3Ls", "raw", 0.0)],     # the same encoders on the exact DINOv2 input (224 px)
        # post hoc (chosen after the Phase 6 dev scores were seen; descriptive only)
        "l14": [("L14", "raw", 0.0)],                             # DINOv2 L/14 alone
        "d3L": [("D3L", "raw", 0.0)],                             # DINOv3 L/16 alone
        "d3mixB": [("B14", "raw", 0.0), ("D3L", "raw", 0.0)],     # DINOv2 B/14 + DINOv3 L/16
        "d3mixL": [("L14", "raw", 0.0), ("D3L", "raw", 0.0)]}     # DINOv2 L/14 + DINOv3 L/16


def get_cfg(cfg_name):
    if cfg_name == "lock":
        return locked_cfg()
    if cfg_name in SWAP:
        return {"views": SWAP[cfg_name], "spec": locked_cfg()["spec"]}
    return RUN[cfg_name]


def run_task(D, cfg_name, draw, stream, seed, device="cuda"):
    cfg = get_cfg(cfg_name)
    for (vname, _, _) in cfg["views"]:
        for part in vname.split("+"):
            assert part in D["feats"], f"features of view {part} are missing for {D['dev']}"
    out_dir = OUT / cfg_name
    name = f"{D['dev']}_draw{draw}_{stream}_seed{seed}"
    if (out_dir / f"{name}.npz").exists():
        return
    t0 = time.time()
    sid, flag, S, bidx = D["tins"][(str(draw), stream, seed)]
    rows = np.array([D["stream_row"][s] for s in sid])
    dr = D["draws"][str(draw)]
    cand_cal = np.argsort(-dr["cal_sims"], axis=1)[:, :V5["K"]]
    cand_sf = D["cand_stream"][rows]
    views = {}
    for (vname, kind, alpha) in cfg["views"]:
        sup, cal, sf = raw_view(D, draw, vname, rows)
        if kind != "raw":
            f = E5.fit_transform(sup, kind, alpha, device)
            sup, cal, sf = f(sup), f(cal), f(sf)
        st = E5.static5(r5.proto_view, support_dall, sup, cal, sf, cand_cal, cand_sf, V5["n0"], V5["m"])
        views[vname] = (sup, cal, sf, st)
    spec = E5.make_spec(**cfg["spec"])
    t_prep = round(time.time() - t0, 1)
    out, adm, timing = E5.run5(views, bidx, spec, device, diag_oracle=flag if spec["variants"].count("Mora") else None)
    arrays = {"sample_id": np.array(sid), "is_ood": flag, "logS": np.log(S.astype(np.float64)),
              "logM": np.log(D["mcm"][rows]), "bidx": bidx, "wnid": D["wnid"][rows]}
    for v in out:
        for k, x in out[v].items():
            assert not np.isnan(x).any(), (v, k)
            arrays[f"s::{v}::{k}"] = x
        for k, x in adm[v].items():
            arrays[f"a::{v}::{k}"] = x
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_dir / f"{name}.npz")
    tt = {v: [round(float(t[:, 1].sum()), 2), round(float(t[:, 2].sum()), 2)] for v, t in timing.items()}
    print(json.dumps({"cfg": cfg_name, "task": name, "seconds": round(time.time() - t0, 1), "prep_seconds": t_prep, "mem_lp_seconds": tt,
                      "keys": len(arrays), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="dev1")
    ap.add_argument("--cfgs", default="base")
    ap.add_argument("--draws", default="0,1,2,3,4")
    ap.add_argument("--streams", default="near,far")
    ap.add_argument("--seeds", default="123,124,125")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    D = load(a.dev)
    tasks = [(c, int(d), s, int(sd)) for c in a.cfgs.split(",") for d in a.draws.split(",") for s in a.streams.split(",")
             for sd in a.seeds.split(",")]
    for t in tasks[a.worker::a.nworkers]:
        run_task(D, *t)
