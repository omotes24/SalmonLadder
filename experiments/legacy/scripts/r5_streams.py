"""R5 Phase 0: configurable dev streams (stream generator), TINS on them, and the frozen v4 components on them.

A stream is defined by a spec (dict) and materialised as r5/streams/<dev>/<name>/stream.parquet:
  id_n        number of ID arrivals (uniform over the dev ID classes, sampled from id_dev; <= 18,000)
  ood         list of groups {"source": "near"|"far", "n_classes": int|None, "per_class": int|None, "n": int|None}
              near: held-out classes (50 images each); far: OpenImage-O val (no classes; "n" images)
  order       random     : uniform random order
              id_burst   : ID arrivals grouped in runs of one class (run length L = spec["run"]), runs in random
                           order, OOD inserted at random positions
              id_first   : every ID arrival before any OOD
              delayed    : half of the near OOD classes (seeded) appear only in the second half of the stream
              pseudo_burst: every source image is followed by k weak augmentations of itself (k = spec["k"],
                           random crop 60-100% area + flip), for ID and OOD; features of the augmented copies come
                           from --stage augfeat
  batch       batch size (TINS and REPRISE use the same batches)
  seed        seed of sampling and order
Stages: build (CPU) -> augfeat (GPU, pseudo_burst only) -> tins (GPU, per shot draw) -> eval (CPU)
The eval stage computes the same variants as r5_eval.py (static, memory v4 / single entrance, LP, full) and logs
the ID share of the memory over time, ID admission per class and Pr(p <= 0.1) of every p-value for ID arrivals.
No label is used by any decision; labels are stored for evaluation only.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

G = {}


def dev_name():
    return "dev2" if C.WORK.name == "dev2" else "dev1"


def sdir(name):
    return r5.R5 / dev_name() / "streams" / name


EXTRA_PREFIX = "places365/"


def extra_feats(model):
    """(features, {sample_id: row}) of the extra far-OOD source (Places365 val; r5_places.py) for a model stem."""
    import torch

    blob = torch.load(r5.R5 / "extra" / f"places.{model}.pt", map_location="cpu")
    return blob["features"], {s: i for i, s in enumerate(blob["sample_id"])}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------
def build(spec):
    rng = np.random.default_rng([int(spec["seed"]), 31])
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    id_dev = samples[samples.split == "id_dev"]
    classes = np.sort(id_dev.class_idx_id.unique())
    n_id = int(spec["id_n"])
    per = n_id // len(classes)
    extra = n_id - per * len(classes)
    pick = []
    bonus = set(rng.choice(classes, extra, replace=False).tolist()) if extra else set()
    for c in classes:
        rows = id_dev[id_dev.class_idx_id == c].sample_id.values
        k = per + (1 if c in bonus else 0)
        pick += list(rng.choice(rows, k, replace=False))
    id_ids = np.array(pick)
    id_cls = id_dev.set_index("sample_id").loc[id_ids].class_idx_id.values.astype(int)
    ood_ids, ood_cls, ood_src = [], [], []
    for g in spec["ood"]:
        if g["source"] == "near":
            near = samples[samples.split == "near_dev"]
            wn = np.sort(near.wnid.unique())
            chosen = rng.choice(wn, g["n_classes"], replace=False) if g.get("n_classes") else wn
            for w in chosen:
                rows = near[near.wnid == w].sample_id.values
                k = g.get("per_class") or len(rows)
                ood_ids += list(rng.choice(rows, k, replace=False))
                ood_cls += [f"near:{w}"] * k
                ood_src += ["near"] * k
        elif g["source"] == "places":
            meta = pd.read_parquet(r5.R5 / "extra" / "places_meta.parquet")
            meta = meta[meta.dev == dev_name()]
            cl = np.sort(meta.cls.unique())
            chosen = rng.choice(cl, g["n_classes"], replace=False) if g.get("n_classes") else cl
            for c in chosen:
                rows = meta[meta.cls == c].sample_id.values
                k = g.get("per_class") or len(rows)
                ood_ids += list(rng.choice(rows, k, replace=False))
                ood_cls += [f"places:{c}"] * k
                ood_src += ["places"] * k
        else:
            far = samples[samples.split == "far_dev"].sample_id.values
            k = g.get("n") or len(far)
            ood_ids += list(rng.choice(far, k, replace=False))
            ood_cls += [f"far:{i}" for i in range(k)]
            ood_src += ["far"] * k
    ood_ids, ood_cls = np.array(ood_ids), np.array(ood_cls)
    items = [(s, False, f"id:{c}", "id") for s, c in zip(id_ids, id_cls)] + \
            [(s, True, c, src) for s, c, src in zip(ood_ids, ood_cls, ood_src)]
    order = spec["order"]
    if order in ("random", "pseudo_burst"):
        seq = [items[i] for i in rng.permutation(len(items))]
    elif order == "id_first":
        seq = [items[i] for i in rng.permutation(len(id_ids))] + \
              [items[len(id_ids) + i] for i in rng.permutation(len(ood_ids))]
    elif order == "id_burst":
        run = int(spec.get("run", 10))
        runs = []
        for c in np.unique(id_cls):
            idx = rng.permutation(np.flatnonzero(id_cls == c))
            runs += [list(idx[i:i + run]) for i in range(0, len(idx), run)]
        runs = [runs[i] for i in rng.permutation(len(runs))]
        blocks = [[items[i] for i in r_] for r_ in runs] + [[items[len(id_ids) + i]] for i in range(len(ood_ids))]
        seq = [x for i in rng.permutation(len(blocks)) for x in blocks[i]]
    elif order == "delayed":
        near_cls = np.unique([c for c in ood_cls if c.startswith("near:")])
        late = set(rng.choice(near_cls, len(near_cls) // 2, replace=False).tolist())
        early_items = [it for it in items if not (it[1] and it[2] in late)]
        late_items = [it for it in items if it[1] and it[2] in late]
        n = len(items)
        base = [early_items[i] for i in rng.permutation(len(early_items))]
        half = n // 2
        second = base[half - 0:] if len(base) > half else []
        first = base[:half]
        mixed = second + late_items
        seq = first + [mixed[i] for i in rng.permutation(len(mixed))]
    else:
        raise ValueError(order)
    rows = []
    k_aug = int(spec.get("k", 0)) if order == "pseudo_burst" else 0
    for s, is_ood, c, src in seq:
        rows.append({"sample_id": s, "src_id": s, "aug": -1, "is_ood": is_ood, "cls": c, "source": src})
        for j in range(k_aug):
            rows.append({"sample_id": f"aug{j}|{s}", "src_id": s, "aug": j, "is_ood": is_ood, "cls": c, "source": src})
    frame = pd.DataFrame(rows)
    frame["position"] = np.arange(len(frame))
    frame["batch_index"] = frame.position // int(spec["batch"])
    out = sdir(spec["name"])
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "stream.parquet", index=False)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    return frame


# ----------------------------------------------------------------------------------------------------------
# augmented features (pseudo bursts)
# ----------------------------------------------------------------------------------------------------------
def augfeat(name):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from vins.features import dino_transform, guard_from_seal, load_dino
    from vins.tins_dev import import_tins, load_clip, make_args

    frame = pd.read_parquet(sdir(name) / "stream.parquet")
    aug = frame[frame.aug >= 0]
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet").set_index("sample_id")
    paths = samples.loc[aug.src_id].path.values
    seeds = [int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) for s in aug.sample_id]
    guard = guard_from_seal()

    class AugSet(Dataset):
        def __init__(self, post):
            self.post = post

        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(guard.check(paths[i])).convert("RGB")
            g = torch.Generator().manual_seed(seeds[i])
            top, left, h, w = _params(img, g)               # random crop of 60-100% of the area, ratio 3/4-4/3
            img = img.crop((left, top, left + w, top + h))
            if torch.rand(1, generator=g).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return self.post(img), i

    def _params(img, g):
        W, H = img.size
        for _ in range(20):
            area = W * H * (0.6 + 0.4 * torch.rand(1, generator=g).item())
            logr = np.log(3 / 4) + (np.log(4 / 3) - np.log(3 / 4)) * torch.rand(1, generator=g).item()
            ar = float(np.exp(logr))
            w, h = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
            if 0 < w <= W and 0 < h <= H:
                top = int(torch.randint(0, H - h + 1, (1,), generator=g))
                left = int(torch.randint(0, W - w + 1, (1,), generator=g))
                return top, left, h, w
        return 0, 0, H, W

    def run(post, fn, batch):
        out, order = [], []
        with torch.no_grad():
            for x, i in DataLoader(AugSet(post), batch_size=batch, num_workers=8, shuffle=False):
                f = fn(x.cuda()).float()
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
                order.append(i)
        assert torch.equal(torch.cat(order), torch.arange(len(paths)))
        return torch.cat(out)

    t = import_tins()
    args = make_args(t, sdir(name) / "clip_cache", "vins_r5_aug")
    net, preprocess = load_clip(t, args)
    feats = {"clip": run(preprocess, net.encode_image, 256)}
    del net
    base = load_dino()
    feats["dino"] = run(dino_transform(), lambda x: base.forward_features(x)["x_norm_clstoken"], 128)
    del base
    large = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    large.load_state_dict(torch.load(C.HOME / ".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
                                     map_location="cpu"), strict=True)
    large = large.eval().cuda()
    feats["dino_vitl14"] = run(dino_transform(), lambda x: large.forward_features(x)["x_norm_clstoken"], 64)
    torch.save({"sample_id": aug.sample_id.tolist(), **feats}, sdir(name) / "augfeat.pt")


# ----------------------------------------------------------------------------------------------------------
# TINS on a custom stream (features from features/clip.pt, as run_tins.py --features ours)
# ----------------------------------------------------------------------------------------------------------
def tins(name, draws, record_cal=False):
    import torch

    from vins.features import load_features
    from vins.tins_dev import Recorder, import_tins, load_clip, make_args, run_stream, setup_to_device

    draws = [d for d in draws if not (sdir(name) / f"tins_draw{d}.npz").exists()]   # a retry resumes
    if not draws:
        print(json.dumps({"skip_existing": name}), flush=True)
        return
    frame = pd.read_parquet(sdir(name) / "stream.parquet")
    clip, blob = load_features(C.FEATURES_DIR / "clip.pt")
    pos = {s: i for i, s in enumerate(blob["sample_id"])}
    feats = torch.zeros((len(frame), clip.shape[1]), dtype=clip.dtype)
    base_rows = frame.aug.values < 0
    ext_rows = frame.sample_id.str.startswith(EXTRA_PREFIX).values
    dev_rows = base_rows & ~ext_rows
    feats[torch.from_numpy(np.flatnonzero(dev_rows))] = clip[[pos[s] for s in frame.sample_id[dev_rows]]]
    if ext_rows.any():
        xf, xpos = extra_feats("clip")
        feats[torch.from_numpy(np.flatnonzero(ext_rows))] = xf[[xpos[s] for s in frame.sample_id[ext_rows]]].to(clip.dtype)
    if (~base_rows).any():
        ab = torch.load(sdir(name) / "augfeat.pt", map_location="cpu")
        apos = {s: i for i, s in enumerate(ab["sample_id"])}
        feats[torch.from_numpy(np.flatnonzero(~base_rows))] = ab["clip"][[apos[s] for s in frame.sample_id[~base_rows]]].to(
            clip.dtype)
    t = import_tins()
    args = make_args(t, sdir(name) / "tins_cache", f"vins_r5_stream_{name}")
    args.batch_size = int(json.loads((sdir(name) / "spec.json").read_text())["batch"])
    net, _ = load_clip(t, args)
    if record_cal:                    # per-batch TINS scores of the draw's calibration shots (as r5_tins.py)
        T = load_script("r5_tins")
        orig = t.compute_grouped_positive_score
        capture = {"last": None}

        def wrapped(**kw):
            capture["last"] = kw
            return orig(**kw)

        t.compute_grouped_positive_score = wrapped
        shot_clip, sblob = load_features(r5.R5 / "features" / "shots.clip.pt")
        spos = {s: i for i, s in enumerate(sblob["sample_id"])}
        dev_clip = clip
    for draw in draws:
        setup = torch.load(r5.R5 / dev_name() / "tins_setup" / f"draw{draw}.pt", map_location="cpu") if draw != "orig" \
            else torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")
        dev_setup = setup_to_device(setup)
        extra = {}
        if record_cal:
            cids = T.calib_ids(draw)
            cal = (dev_clip[[pos[s] for s in cids]] if draw == "orig" else shot_clip[[spos[s] for s in cids]])
            rec = T.CalRecorder(capture, cal.to(dev_setup["positive_features"].device), orig)
        else:
            rec = Recorder()
        t.setup_seed(args.seed)
        tick = time.time()
        scores = run_stream(t, args, net, dev_setup, feats, hook=rec)
        per = rec.per_sample(len(frame))
        assert np.array_equal(per["S_final"], scores)
        assert (per["batch_index"] == frame.batch_index.values).all(), "TINS batches != stream batches"
        tmp = sdir(name) / f"tins_draw{draw}.tmp.npz"
        if record_cal:
            extra = {"cal_scores": np.stack(rec.cal_scores).astype(np.float32), "cal_ids": cids}
        np.savez_compressed(tmp, sample_id=frame.sample_id.values, S_final=per["S_final"],
                            batch_index=per["batch_index"], **extra)
        os.replace(tmp, sdir(name) / f"tins_draw{draw}.npz")
        print(json.dumps({"tins": f"{name}/draw{draw}", "seconds": round(time.time() - tick, 1)}), flush=True)


# ----------------------------------------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------------------------------------
def load_r5_eval():
    spec = importlib.util.spec_from_file_location("r5_eval", ROOT / "scripts" / "r5_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def views_for(name, draw):
    """REPRISE views whose queries are the calibration shots + the stream (augmented copies included)."""
    import torch

    from vins.features import load_features

    E = load_r5_eval()
    ev = E.load_ev()
    frame = pd.read_parquet(sdir(name) / "stream.parquet")
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    n_id = len(id_classes)
    sup_ids, cal_ids, src = E.shot_table(draw, id_classes, samples)
    pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
    aug = None
    if (frame.aug >= 0).any():
        aug = torch.load(sdir(name) / "augfeat.pt", map_location="cpu")
        apos = {s: i for i, s in enumerate(aug["sample_id"])}
    base_rows = frame.aug.values < 0
    ext_rows = frame.sample_id.str.startswith(EXTRA_PREFIX).values
    dev_rows = base_rows & ~ext_rows
    k_stream = np.zeros((len(frame), C.K_TOP), dtype=np.int64)
    k_stream[dev_rows] = np.array(dview.loc[frame.sample_id[dev_rows]].K_id.tolist())
    if ext_rows.any():
        xc, xpos = extra_feats("clip")
        k_stream[ext_rows] = (xc[[xpos[s] for s in frame.sample_id[ext_rows]]].float() @ pos_text.T).topk(
            C.K_TOP, dim=1).indices.numpy()
    if aug is not None:
        ac = aug["clip"][[apos[s] for s in frame.sample_id[~base_rows]]].float()
        k_stream[~base_rows] = (ac @ pos_text.T).topk(C.K_TOP, dim=1).indices.numpy()
    if src == "features":
        k_cal = np.array(dview.loc[cal_ids].K_id.tolist())
    else:
        clip_f, blob = load_features(r5.R5 / "features" / "shots.clip.pt")
        cpos = {s: i for i, s in enumerate(blob["sample_id"])}
        k_cal = (clip_f[[cpos[s] for s in cal_ids]].float() @ pos_text.T).topk(C.K_TOP, dim=1).indices.numpy()
    cand = np.concatenate([k_cal, k_stream])
    views = {}
    for vname, dev_file, shot_file in E.VIEWS:
        dev_f, blob = load_features(C.FEATURES_DIR / dev_file)
        dpos = {s: i for i, s in enumerate(blob["sample_id"])}
        if src == "features":
            sup_f, cal_f = dev_f[[dpos[s] for s in sup_ids]], dev_f[[dpos[s] for s in cal_ids]]
        else:
            sh_f, sblob = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
            spos = {s: i for i, s in enumerate(sblob["sample_id"])}
            sup_f, cal_f = sh_f[[spos[s] for s in sup_ids]], sh_f[[spos[s] for s in cal_ids]]
        st = np.zeros((len(frame), dev_f.shape[1]), dtype=np.float32)
        st[dev_rows] = dev_f[[dpos[s] for s in frame.sample_id[dev_rows]]].numpy()
        if ext_rows.any():
            xf, xpos = extra_feats(shot_file)
            st[ext_rows] = xf[[xpos[s] for s in frame.sample_id[ext_rows]]].float().numpy()
        if aug is not None:
            st[~base_rows] = aug[shot_file][[apos[s] for s in frame.sample_id[~base_rows]]].numpy()
        support = sup_f.numpy().astype(np.float32).reshape(n_id, 12, -1)
        q = np.concatenate([cal_f.numpy().astype(np.float32), st])
        is_cal = np.zeros(len(q), dtype=bool)
        is_cal[:len(cal_ids)] = True
        v = ev.custom_view(support, q, cand, is_cal, np.repeat(np.arange(n_id), 4), m=2, proto=True)
        v["support_arr"], v["q"] = v["support"], q
        views[vname] = v
    return views, len(cal_ids), frame


def eval_task(job):
    name, draw = job
    from vins.metrics import measures as upstream

    views, n_cal, frame = views_for(name, draw)
    z = np.load(sdir(name) / f"tins_draw{draw}.npz", allow_pickle=True)
    S, bidx = z["S_final"].astype(np.float64), z["batch_index"]
    is_ood = frame.is_ood.values.astype(bool)
    q = n_cal + np.arange(len(frame))
    arr, logs = {"S": S}, {}
    for vname, v in views.items():
        sf, d, p_all = v["q"][q], v["d"][q], v["p_all"][q]
        med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
        m3 = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, r5.V4_ENTRANCE)
        pt3, _ = r5.memory_p(sf, d, bidx, m3[-1], v["cal_feats"], v["d_cal"], med, mad)
        m1 = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, (0.10,))
        pt1, _ = r5.memory_p(sf, d, bidx, m1[-1], v["cal_feats"], v["d_cal"], med, mad)
        lp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, **r5.V4_LP)
        arr.update({f"p_{vname}": v["p"][q], f"pt3_{vname}": pt3, f"pt1_{vname}": pt1, f"plp_{vname}": lp["p"]})
        M = m3[-1]
        nb = int(bidx.max()) + 1
        cum_id = np.cumsum(np.bincount(bidx[M & ~is_ood], minlength=nb))
        cum_all = np.cumsum(np.bincount(bidx[M], minlength=nb))
        idc = frame.cls.values
        id_rows = ~is_ood
        per_cls = pd.Series(M[id_rows]).groupby(idc[id_rows]).mean()
        logs[vname] = {"memory_id_share_by_batch": (cum_id / np.maximum(cum_all, 1)).tolist(),
                       "id_admission": float(M[id_rows].mean()), "ood_admission": float(M[is_ood].mean()),
                       "id_admission_per_class_q": [float(x) for x in per_cls.quantile([0.5, 0.9, 0.99, 1.0])],
                       "pr_le_0.1": {k: float((arr[f"{k}_{vname}"][id_rows] <= 0.1).mean())
                                     for k in ("p", "pt3", "pt1", "plp")}}
    P = lambda k: arr[f"{k}_B14"] * arr[f"{k}_L14"]  # noqa: E731
    vis = {"static": P("p"), "mem": P("pt3"), "mem1": P("pt1"), "lp": P("plp"), "full": P("pt3") * P("plp"),
           "single_lp": P("pt1") * P("plp")}
    scores = {"tins": S}
    for k, v in vis.items():
        scores[f"v_{k}"], scores[f"s_{k}"] = v, S * v
    if "cal_scores" in z.files:       # M3: conformalised TINS score and combination rules (as r5_mods.py)
        from scipy.stats import chi2

        cal_s = z["cal_scores"].astype(np.float64)
        pT = np.empty(len(S))
        for bi, b in enumerate(np.unique(bidx)):
            rows = np.flatnonzero(bidx == b)
            pT[rows] = r5.pval_low(cal_s[bi], S[rows])
        vp = [arr["pt3_B14"], arr["pt3_L14"], arr["plp_B14"], arr["plp_L14"]]
        for rule in ("product", "cauchy", "hmp"):
            scores[f"m3_{rule}"] = r5.combine_p([pT] + vp, rule)
            scores[f"m3_vis_{rule}"] = r5.combine_p(vp, rule)
        per_view = [chi2.sf(-2 * np.log(np.clip(arr[f"pt3_{n}"] * arr[f"plp_{n}"], 1e-300, 1)), 4) for n in ("B14", "L14")]
        scores["m3_withinview_product"] = r5.combine_p([pT] + per_view, "product")
        scores["m3_withinview_cauchy"] = r5.combine_p([pT] + per_view, "cauchy")
        scores["m3_pT_only"] = pT
        arr["pT"] = pT
    metrics = {}
    for k, s in scores.items():
        m = upstream(s[~is_ood], s[is_ood])
        metrics[k] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
    out = sdir(name) / f"eval_draw{draw}.json"
    out.write_text(json.dumps({"metrics": metrics, "logs": logs, "n": len(frame), "n_ood": int(is_ood.sum())}) + "\n")
    np.savez_compressed(sdir(name) / f"eval_draw{draw}.npz", sample_id=frame.sample_id.values, is_ood=is_ood,
                        cls=frame.cls.values, batch_index=bidx,
                        **{k: np.asarray(v, dtype=np.float32) for k, v in arr.items()})
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "augfeat", "tins", "eval"], required=True)
    parser.add_argument("--specs", required=True, help="json file with a list of specs")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--draws", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--record-cal", action="store_true", help="tins: also score the calibration shots per batch")
    opts = parser.parse_args()
    specs = [s for s in json.loads(Path(opts.specs).read_text()) if s.get("dev", "dev1") == dev_name()]
    if opts.names:
        specs = [s for s in specs if s["name"] in opts.names]
    if opts.stage == "build":
        for s in specs:
            f = build(s)
            print(json.dumps({"built": s["name"], "n": len(f), "n_ood": int(f.is_ood.sum())}), flush=True)
    elif opts.stage == "augfeat":
        for s in specs:
            if s["order"] == "pseudo_burst" and not (sdir(s["name"]) / "augfeat.pt").exists():
                augfeat(s["name"])
    elif opts.stage == "tins":
        for s in specs:
            tins(s["name"], opts.draws, opts.record_cal)
    else:
        from vins.tins_dev import import_tins

        import_tins()
        jobs = [(s["name"], d) for s in specs for d in opts.draws]
        with mp.get_context("fork").Pool(opts.workers) as pool:
            for j in pool.imap_unordered(eval_task, jobs):
                print(json.dumps({"eval": j}), flush=True)


if __name__ == "__main__":
    main()
