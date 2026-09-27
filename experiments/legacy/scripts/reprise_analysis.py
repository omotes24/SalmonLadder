"""Post-hoc analyses of the frozen REPRISE (= iter3/frozen_m3.json). Nothing here selects or changes the method.

--part openood : OpenOOD v1.5 ImageNet-1K test streams (5 datasets x stream orders 123..127)
--part fourood : Four-OOD (ImageNet-1K val 50,000 vs iNaturalist / SUN / Places / DTD), TINS stream orders 123..125
Per (stream, order) and view (B14, L14), saved per sample:
  p      static prototype-view p-value (no memory, no propagation)
  pt1    memory p-value with a single-stage entrance (p_all <= 0.10)        [ablation]
  pt3    memory p-value with REPRISE's iterated entrance (0.40 / 0.30 / 0.10)
  plp    conformal label-propagation p-value (k = 10, alpha = 0.9, gamma = 3, 15 iterations)
  adm1 / adm3 admission masks, plus wall-clock seconds of each part.
Output: <WORK>/analysis/reprise/<part>/<stream>_seed<s>.npz
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402

RES = C.WORK / "test_eval" / "results"
FEAT = C.WORK / "extra_feats"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
G = {}


def load_ev():
    frozen = json.loads((C.WORK / "iter3" / "frozen_m3.json").read_text())
    code = ROOT / "scripts" / "iter3_eval.py"
    assert hashlib.sha256(code.read_bytes()).hexdigest() == frozen["code"]["sha256"], "iter3_eval.py changed"
    spec = importlib.util.spec_from_file_location("iter3_eval", code)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev, frozen["config"]


def topk(clip_feats, pos, k=5, chunk=8192):
    pos = pos.float()
    return torch.cat([(clip_feats[lo:lo + chunk].float() @ pos.T).topk(k, dim=1).indices
                      for lo in range(0, len(clip_feats), chunk)]).numpy()


def shots():
    b = torch.load(RES / "features.pt", map_location="cpu")
    l = torch.load(RES / "features_vitl14.pt", map_location="cpu")
    assert b["paths"] == l["paths"]
    return b, {"B14": b["dino"].numpy().astype(np.float32), "L14": l["dino"].numpy().astype(np.float32)}


def make_views(ev, feats_shots, eval_feats, cand):
    """feats_shots: first 12,000 support + 4,000 calibration rows; eval_feats: the evaluated images (per view)."""
    n_sup, n_cal = 12000, 4000
    cal_cls = np.repeat(np.arange(1000), 4)
    views, queries = {}, {}
    for name in ("B14", "L14"):
        f = feats_shots[name]
        support = f[:n_sup].reshape(1000, 12, -1)
        q = np.concatenate([f[n_sup:n_sup + n_cal], eval_feats[name]]).astype(np.float32)
        is_cal = np.zeros(len(q), dtype=bool)
        is_cal[:n_cal] = True
        v = ev.custom_view(support, q, cand, is_cal, cal_cls, m=2, proto=True)
        v["support_arr"] = v["support"]
        views[name], queries[name] = v, q
    return views, queries


def run_one(task):
    stream, seed, path = task
    ev, cfg = G["ev"], G["cfg"]
    k_, a_, g_ = cfg["lp"].split(",")
    run = np.load(path)
    ids, is_ood = run["sample_id"], run["is_ood"].astype(bool)
    S, bidx = run["S_final"].astype(np.float64), run["batch_index"]
    q = np.array([G["row"][x] for x in ids])
    arrays, times = {}, {}
    for name in ("B14", "L14"):
        v, sf = G["views"][name], G["queries"][name][q]
        d, p_all = v["d"][q], v["p_all"][q]
        t0 = time.time()
        adm1 = p_all <= 0.10
        pt1, _ = ev.p_memory(v, sf, d, bidx, adm1)
        t1 = time.time()
        adm3 = ev.admit_mask(v, sf, d, p_all, bidx, cfg["rule"])
        pt3, _ = ev.p_memory(v, sf, d, bidx, adm3)
        t2 = time.time()
        plp = ev.lp_pvalues(v, sf, bidx, int(k_), float(a_), float(g_))
        t3 = time.time()
        arrays.update({f"p_{name}": v["p"][q].astype(np.float32), f"pt1_{name}": pt1.astype(np.float32),
                       f"pt3_{name}": pt3.astype(np.float32), f"plp_{name}": plp.astype(np.float32),
                       f"adm1_{name}": adm1, f"adm3_{name}": adm3})
        times[name] = {"memory_single_s": t1 - t0, "memory_reprise_s": t2 - t1, "lp_s": t3 - t2}
    out = G["out"] / f"{stream}_seed{seed}.npz"
    np.savez_compressed(out, sample_id=ids, is_ood=is_ood, S=S, batch_index=bidx, times=json.dumps(times), **arrays)
    return stream, seed, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["openood", "fourood"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    opts = parser.parse_args()
    start = time.time()
    ev, cfg = load_ev()
    b, feats = shots()
    out = C.WORK / "analysis" / "reprise" / opts.part
    out.mkdir(parents=True, exist_ok=True)
    if opts.part == "openood":
        n = 16000
        views, queries = make_views(ev, feats, {k: v[n:] for k, v in feats.items()}, np.asarray(b["cand"]))
        # make_views prepends the calibration rows again; the evaluated rows start at 4000 + (index in test list)
        test_ids = sorted({x for ds in OO for x in np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist()})
        assert len(test_ids) == len(feats["B14"]) - n
        row = {s: 4000 + i for i, s in enumerate(test_ids)}
        cand_ok = np.asarray(b["cand"])
        assert len(cand_ok) == 4000 + len(test_ids)
        tasks = [(ds, s, C.WORK / "test_runs" / "default" / f"{ds}_seed{s}.npz")
                 for ds in ("ssb_hard", "openimageo", "inaturalist", "ninco", "textures") for s in (123, 124, 125, 126, 127)]
    else:
        names = ["in_val"] + FOUR
        dn = {n_: torch.load(FEAT / f"{n_}.dino.pt", map_location="cpu") for n_ in names}
        dl = {n_: torch.load(FEAT / f"{n_}.dinol14.pt", map_location="cpu") for n_ in names}
        cl = {n_: torch.load(FEAT / f"{n_}.clipb16.pt", map_location="cpu") for n_ in names}
        ids = sum((dn[n_]["ids"] for n_ in names), [])
        assert ids == sum((dl[n_]["ids"] for n_ in names), []) == sum((cl[n_]["ids"] for n_ in names), [])
        pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"]
        cand = np.concatenate([np.asarray(b["cand"])[:4000], topk(torch.cat([cl[n_]["features"] for n_ in names]), pos)])
        ev_feats = {"B14": torch.cat([dn[n_]["features"] for n_ in names]).numpy(),
                    "L14": torch.cat([dl[n_]["features"] for n_ in names]).numpy()}
        views, queries = make_views(ev, feats, ev_feats, cand)
        row = {s: 4000 + i for i, s in enumerate(ids)}
        seeds = sorted({int(p.stem.split("seed")[1]) for p in (C.WORK / "extra_runs" / "fourood").glob("imagenet_inat_seed*.npz")})
        tasks = [(ood, s, C.WORK / "extra_runs" / "fourood" / f"imagenet_{ood}_seed{s}.npz") for ood in FOUR for s in seeds]
    G.update({"ev": ev, "cfg": cfg, "views": views, "queries": queries, "row": row, "out": out})
    print(json.dumps({"part": opts.part, "tasks": len(tasks), "setup_s": round(time.time() - start, 1)}), flush=True)
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for stream, seed, times in pool.imap_unordered(run_one, tasks):
            print(json.dumps({"stream": stream, "seed": seed, "times": {k: {kk: round(vv, 1) for kk, vv in v.items()}
                                                                       for k, v in times.items()}}), flush=True)
    print(json.dumps({"done": opts.part, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
