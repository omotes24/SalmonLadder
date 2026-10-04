"""Phase 3: E2-E4 stream conditions on the OpenOOD v1.5 ImageNet-1K test data (run once, pre-registered).

Sources: ID = test_imagenet images; near OOD = SSB-hard (classes = image folders) and NINCO; far OOD = iNaturalist,
Textures, OpenImage-O. Per-image TINS features come from the unchanged-upstream per-dataset caches (as
run_tins_test.py); DINOv2 features and CLIP candidates from test_eval/results (the p3_eval row space). Specs (seeded):
  E2_ratio{r}   ID 9,000 + SSB-hard images at OOD ratio r in {1, 2, 5, 10, 25, 50}%
  E2_cls{c}x{n} ID 9,000 + 980 SSB-hard images as c classes x n images (980x1, 196x5, 49x20, 20x49)
  E3_idburst    ID in runs of 10 images of one class, 1,000 SSB-hard images at random positions
  E3_idfirst    all ID before any OOD (ID 9,000, SSB-hard 1,000)
  E3_pseudo     2,000 ID + 200 SSB-hard sources, each followed by 4 weak augmentations of itself (crop 60-100% area,
                flip; augmented features extracted in Phase 3 from the sealed images)
  E3_mixall     ID 9,000 + 500 each of SSB-hard, NINCO, iNaturalist, Textures, OpenImage-O
  E4_delayed    ID 9,000 + SSB-hard 100 classes x 10; half of the classes appear only in the second half
Each spec is run with order seeds 123, 124, 125 (TINS re-run on the stream's own order and batches).
--stage build | augfeat | tins | eval ; output <R5>/phase3/streams/<name>/
"""
import argparse
import glob
import hashlib
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
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
OUT = P3 / "streams"
GROUP = {"ssb_hard": "nearood", "ninco": "nearood", "inaturalist": "farood", "textures": "farood", "openimageo": "farood"}
SEEDS = (123, 124, 125)
SPECS = [{"name": f"E2_ratio{r}", "id_n": 9000, "ood": [("ssb_hard", None, None, int(round(9000 * r / (100 - r))))],
          "order": "random"} for r in (1, 2, 5, 10, 25, 50)]
SPECS += [{"name": f"E2_cls{c}x{n}", "id_n": 9000, "ood": [("ssb_hard", c, n, None)], "order": "random"}
          for c, n in ((980, 1), (196, 5), (49, 20), (20, 49))]
SPECS += [{"name": "E3_idburst", "id_n": 9000, "ood": [("ssb_hard", None, None, 1000)], "order": "id_burst", "run": 10},
          {"name": "E3_idfirst", "id_n": 9000, "ood": [("ssb_hard", None, None, 1000)], "order": "id_first"},
          {"name": "E3_pseudo", "id_n": 2000, "ood": [("ssb_hard", None, None, 200)], "order": "pseudo", "k": 4},
          {"name": "E3_mixall", "id_n": 9000, "ood": [(d, None, None, 500) for d in ("ssb_hard", "ninco", "inaturalist",
                                                                                   "textures", "openimage_o")],
           "order": "random"},
          {"name": "E4_delayed", "id_n": 9000, "ood": [("ssb_hard", 100, 10, None)], "order": "delayed"}]
G = {}


def test_table():
    """sorted OpenOOD test ids, their paths and 'classes' (ID: ImageNet label via the path; OOD: parent folder)."""
    b = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    ids = sorted({x for ds in GROUP for x in np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist()})
    paths = b["paths"][16000:]
    assert len(paths) == len(ids)
    ds = [x.rsplit("_", 1)[0] for x in ids]
    lab = {}
    for line in (C.OPENOOD_IMGLIST_DIR / "test_imagenet.txt").read_text().splitlines():
        if line.strip():
            rel, y = line.split()
            lab[Path(rel).name] = f"in{int(y):03d}"
    cls = [lab[Path(p).name] if d == "imagenet" else Path(p).parent.name for p, d in zip(paths, ds)]
    return pd.DataFrame({"sample_id": ids, "path": paths, "ds": ds, "cls": cls})


def build(spec, seed):
    tab = G["tab"]
    rng = np.random.default_rng([20260927, 91, int(hashlib.sha256(spec["name"].encode()).hexdigest()[:8], 16), seed])
    idp = tab[tab.ds == "imagenet"]
    id_rows = idp.sample(n=spec["id_n"], random_state=int(rng.integers(2**31))) if spec["order"] != "id_burst" else None
    if spec["order"] == "id_burst":                               # 900 classes x 10 consecutive images
        cls = rng.choice(idp.cls.unique(), spec["id_n"] // spec["run"], replace=False)
        parts = [idp[idp.cls == c].sample(n=spec["run"], random_state=int(rng.integers(2**31))) for c in cls]
        id_rows = pd.concat(parts)
    ood = []
    for ds, n_cls, per, n in spec["ood"]:
        pool = tab[tab.ds == ds]
        if n_cls:
            cl = rng.choice(pool.cls.unique(), n_cls, replace=False)
            for c in cl:
                sub = pool[pool.cls == c]
                ood.append(sub.sample(n=min(per, len(sub)), random_state=int(rng.integers(2**31))))
        else:
            ood.append(pool.sample(n=n, random_state=int(rng.integers(2**31))))
    ood = pd.concat(ood)
    id_items = list(zip(id_rows.sample_id, [False] * len(id_rows), id_rows.cls))
    ood_items = list(zip(ood.sample_id, [True] * len(ood), ood.ds + ":" + ood.cls))
    order = spec["order"]
    if order in ("random", "pseudo"):
        items = id_items + ood_items
        seq = [items[i] for i in rng.permutation(len(items))]
    elif order == "id_first":
        seq = [id_items[i] for i in rng.permutation(len(id_items))] + [ood_items[i] for i in rng.permutation(len(ood_items))]
    elif order == "id_burst":
        runs = [id_items[i:i + spec["run"]] for i in range(0, len(id_items), spec["run"])]
        blocks = runs + [[x] for x in ood_items]
        seq = [x for i in rng.permutation(len(blocks)) for x in blocks[i]]
    elif order == "delayed":
        ocls = sorted({c for _, _, c in ood_items})
        late = set(rng.choice(ocls, len(ocls) // 2, replace=False).tolist())
        early = [x for x in id_items + ood_items if not (x[1] and x[2] in late)]
        early = [early[i] for i in rng.permutation(len(early))]
        half = (len(id_items) + len(ood_items)) // 2
        second = early[half:] + [x for x in ood_items if x[2] in late]
        seq = early[:half] + [second[i] for i in rng.permutation(len(second))]
    rows = []
    for s, o, c in seq:
        rows.append({"sample_id": s, "src_id": s, "aug": -1, "is_ood": o, "cls": c})
        for j in range(spec.get("k", 0) if order == "pseudo" else 0):
            rows.append({"sample_id": f"aug{j}|{s}", "src_id": s, "aug": j, "is_ood": o, "cls": c})
    f = pd.DataFrame(rows)
    f["position"] = np.arange(len(f))
    f["batch_index"] = f.position // 256
    return f


def tins_features(frame):
    """TINS-path CLIP features of every stream row (per-dataset upstream caches; augmented copies from augfeat)."""
    if "clip_table" not in G:
        codex = C.CODEX_TINS
        table = {}
        for ds in GROUP:
            z = np.load(codex / f"scores_openood_{GROUP[ds]}_{ds}.npz", allow_pickle=True)
            path = glob.glob(str(codex / "cache" / "image_features" / f"imgfeat_openood_{GROUP[ds]}_{ds}_stream_*.pt"))
            blob = torch.load(path[0], map_location="cpu")
            for s, f in zip(z["sample_id"].astype(str), blob["image_features"]):
                table[s] = f
        G["clip_table"] = table
    feats = []
    aug = None
    for s, a in zip(frame.sample_id, frame.aug):
        if a < 0:
            feats.append(G["clip_table"][s])
        else:
            if aug is None:
                aug = torch.load(OUT / "augfeat.pt", map_location="cpu")
                apos = {x: i for i, x in enumerate(aug["sample_id"])}
            feats.append(aug["clip"][apos[s]].to(G["clip_table"][frame.sample_id.iloc[0]].dtype))
    return torch.stack(feats)


def stage_build():
    G["tab"] = test_table()
    for spec in SPECS:
        for seed in SEEDS:
            d = OUT / spec["name"]
            d.mkdir(parents=True, exist_ok=True)
            build(spec, seed).to_parquet(d / f"stream_seed{seed}.parquet", index=False)
    print(json.dumps({"built": len(SPECS) * len(SEEDS)}))


def stage_augfeat(unseal):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from vins.features import dino_transform, load_dino
    from vins.tins_dev import import_tins, load_clip, make_args

    digest = hashlib.sha256((P3 / "prereg_phase3.json").read_bytes()).hexdigest()
    if unseal != digest:
        raise SystemExit("the unseal token does not match the frozen Phase 3 pre-registration")
    tab = test_table().set_index("sample_id")
    frames = [pd.read_parquet(p) for p in sorted((OUT / "E3_pseudo").glob("stream_seed*.parquet"))]
    aug = pd.concat([f[f.aug >= 0] for f in frames]).drop_duplicates("sample_id")
    paths = tab.loc[aug.src_id].path.values
    seeds = [int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) for s in aug.sample_id]

    class A(Dataset):
        def __init__(self, post):
            self.post = post

        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            g = torch.Generator().manual_seed(seeds[i])
            W, H = img.size
            box = (0, 0, W, H)
            for _ in range(20):
                area = W * H * (0.6 + 0.4 * torch.rand(1, generator=g).item())
                lr = np.log(3 / 4) + (np.log(4 / 3) - np.log(3 / 4)) * torch.rand(1, generator=g).item()
                w, h = int(round(np.sqrt(area * np.exp(lr)))), int(round(np.sqrt(area / np.exp(lr))))
                if 0 < w <= W and 0 < h <= H:
                    top = int(torch.randint(0, H - h + 1, (1,), generator=g))
                    left = int(torch.randint(0, W - w + 1, (1,), generator=g))
                    box = (left, top, left + w, top + h)
                    break
            img = img.crop(box)
            if torch.rand(1, generator=g).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return self.post(img), i

    @torch.no_grad()
    def run(post, fn, bs):
        out, order = [], []
        for x, i in DataLoader(A(post), batch_size=bs, num_workers=8, shuffle=False):
            f = fn(x.cuda()).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
            order.append(i)
        assert torch.equal(torch.cat(order), torch.arange(len(paths)))
        return torch.cat(out)

    t = import_tins()
    args = make_args(t, OUT / "clip_cache", "vins_p3_aug")
    net, preprocess = load_clip(t, args)
    feats = {"clip": run(preprocess, net.encode_image, 256)}
    del net
    base = load_dino()
    feats["B14"] = run(dino_transform(), lambda x: base.forward_features(x)["x_norm_clstoken"], 128)
    base = None
    large = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    large.load_state_dict(torch.load(C.HOME / ".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
                                     map_location="cpu"), strict=True)
    large = large.eval().cuda()
    feats["L14"] = run(dino_transform(), lambda x: large.forward_features(x)["x_norm_clstoken"], 64)
    torch.save({"sample_id": aug.sample_id.tolist(), **feats, "prereg_sha256": digest}, OUT / "augfeat.pt")
    print(json.dumps({"augmented": len(paths)}))


def stage_tins(names):
    rt = __import__("importlib.util").util
    spec_ = rt.spec_from_file_location("run_tins_test", ROOT / "scripts" / "run_tins_test.py")
    RT = rt.module_from_spec(spec_)
    spec_.loader.exec_module(RT)
    from vins.tins_dev import Recorder, get_logger, import_tins, load_clip

    t = import_tins()
    cache = C.WORK / "test_runs" / "tins_cache"
    args = RT.codex_args(t, cache, "vins_p3_streams", 123)
    t.setup_seed(args.seed)
    log = get_logger(OUT / "tins.log")
    net, preprocess = load_clip(t, args)
    device = next(net.parameters()).device
    protos = t.load_or_build_class_prototypes(args, net, preprocess, log)[0].to(device)
    labels = [str(x) for x in t.get_test_labels(args, None)]
    pos = t.encode_texts(net, [args.pos_prompt.format(x) for x in labels], batch_size=args.text_batch_size,
                         device=device, desc="pos").to(device)
    base_sim = (pos * protos).sum(dim=1)
    neg, _, words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=labels, positive_features=pos,
                                                     class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, words, protos, device, log)
    for spec in SPECS:
        if names and spec["name"] not in names:
            continue
        for seed in SEEDS:
            d = OUT / spec["name"]
            target = d / f"tins_seed{seed}.npz"
            if target.exists():
                continue
            frame = pd.read_parquet(d / f"stream_seed{seed}.parquet")
            feats = tins_features(frame)
            rec = Recorder()
            args.stream_seed = seed
            t.setup_seed(args.seed)
            scores = t.compute_tins_scores_from_image_features(
                image_features=feats, args=args, model=net, positive_features=pos, negative_features=neg.to(device),
                inversion_init_candidates=init, class_prototypes=protos, base_sim=base_sim, hook=rec)
            per = rec.per_sample(len(frame))
            assert np.array_equal(per["S_final"], scores)
            assert (per["batch_index"] == frame.batch_index.values).all()
            np.savez_compressed(target, sample_id=frame.sample_id.values, S_final=per["S_final"],
                                batch_index=per["batch_index"])
            print(json.dumps({"tins": f"{spec['name']}/{seed}"}), flush=True)


def eval_task(job):
    name, seed = job
    from vins.metrics import measures as upstream

    pr = G["prereg"]
    frame = pd.read_parquet(OUT / name / f"stream_seed{seed}.parquet")
    z = np.load(OUT / name / f"tins_seed{seed}.npz", allow_pickle=True)
    S, bidx = z["S_final"].astype(np.float64), z["batch_index"]
    is_ood = frame.is_ood.values.astype(bool)
    sc, logs = {"tins": S}, {}
    for tag in ("v4", "v5"):
        cfg = pr["configs"][tag]
        res = {}
        for vn in ("B14", "L14"):
            v = G["views"][(tag, vn)]
            q = G["qrows"](frame, vn, v)
            sf, d, p_all = q["sf"], q["d"], q["p_all"]
            med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
            M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, tuple(cfg["thresholds"]), cfg["m"])[-1]
            pt, _ = r5.memory_p(sf, d, bidx, M, v["cal_feats"], v["d_cal"], med, mad, cfg["m"])
            lp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, k=cfg["kg"], alpha=cfg["lam"], gamma=cfg["gamma"],
                           iters=15)["p"]
            res[vn] = (q["p"], pt, lp, M)
        P = lambda i: res["B14"][i] * res["L14"][i]  # noqa: E731
        sc[f"{tag}"], sc[f"{tag}_lp"], sc[f"{tag}_static"], sc[f"{tag}_mem"] = S * P(1) * P(2), S * P(2), S * P(0), S * P(1)
        logs[tag] = {"id_admission": float(np.mean([res[v][3][~is_ood].mean() for v in res])),
                     "ood_admission": float(np.mean([res[v][3][is_ood].mean() for v in res])),
                     "pr_pt_le_0.1_id": float(np.mean([(res[v][1][~is_ood] <= 0.1).mean() for v in res]))}
    metrics = {k: {"AUROC": 100 * upstream(s[~is_ood], s[is_ood])["AUROC"],
                   "FPR95": 100 * upstream(s[~is_ood], s[is_ood])["FPR95"]} for k, s in sc.items()}
    (OUT / name / f"eval_seed{seed}.json").write_text(json.dumps({"metrics": metrics, "logs": logs, "n": len(frame),
                                                                  "n_ood": int(is_ood.sum())}) + "\n")
    return job


def stage_eval(workers):
    from vins.tins_dev import import_tins

    import_tins()
    spec_ = __import__("importlib.util").util.spec_from_file_location("p3_eval", ROOT / "scripts" / "p3_eval.py")
    PE = __import__("importlib.util").util.module_from_spec(spec_)
    spec_.loader.exec_module(PE)
    pr = json.loads((P3 / "prereg_phase3.json").read_text())
    D = PE.load_part("openood")
    aug = torch.load(OUT / "augfeat.pt", map_location="cpu") if (OUT / "augfeat.pt").exists() else None
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float()
    views = {}
    for tag in ("v4", "v5"):
        cfg = pr["configs"][tag]
        for vn in ("B14", "L14"):
            views[(tag, vn)] = PE.view(D, vn, cfg["n0"], cfg["K"], cfg["m"])
            views[(tag, vn)]["K"] = cfg["K"]
            views[(tag, vn)]["n0"], views[(tag, vn)]["m"] = cfg["n0"], cfg["m"]

    def qrows(frame, vn, v):
        """features, d, p_all, p of the stream rows (augmented copies scored with the same view statistics)."""
        base = frame.aug.values < 0
        rows = np.zeros(len(frame), dtype=np.int64)
        rows[base] = [D["row"][s] for s in frame.sample_id[base]]
        sf = np.zeros((len(frame), v["q"].shape[1]), dtype=np.float32)
        sf[base] = v["q"][rows[base]]
        d, p_all, p = np.zeros(len(frame)), np.zeros(len(frame)), np.zeros(len(frame))
        d[base], p_all[base], p[base] = v["d"][rows[base]], v["p_all"][rows[base]], v["p"][rows[base]]
        if (~base).any():
            apos = {x: i for i, x in enumerate(aug["sample_id"])}
            ai = [apos[s] for s in frame.sample_id[~base]]
            fa = aug[vn][ai].float().numpy()
            fa /= np.linalg.norm(fa, axis=1, keepdims=True)
            ca = (aug["clip"][ai].float() @ pos.T).topk(v["K"], dim=1).indices.numpy()
            sup = D["shots"][vn][:12000].reshape(1000, 12, -1)
            is_cal = np.zeros(4000 + len(ai), dtype=bool)
            is_cal[:4000] = True
            qa = np.concatenate([v["cal_feats"], fa])
            va = r5.proto_view(sup, qa, np.concatenate([D["cands"][v["K"]][:4000], ca]), is_cal, n0=v["n0"], m=v["m"])
            sf[~base], d[~base], p_all[~base], p[~base] = fa, va["d"][4000:], va["p_all"][4000:], va["p"][4000:]
        return {"sf": sf, "d": d, "p_all": p_all, "p": p}

    G.update(prereg=pr, views=views, qrows=qrows)
    jobs = [(s["name"], sd) for s in SPECS for sd in SEEDS if (OUT / s["name"] / f"tins_seed{sd}.npz").exists()]
    with mp.get_context("fork").Pool(workers) as pool:
        for j in pool.imap_unordered(eval_task, jobs):
            print(json.dumps({"eval": j}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "augfeat", "tins", "eval"], required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--unseal", default=None)
    parser.add_argument("--workers", type=int, default=12)
    opts = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if opts.stage == "build":
        stage_build()
    elif opts.stage == "augfeat":
        stage_augfeat(opts.unseal)
    elif opts.stage == "tins":
        stage_tins(opts.names)
    else:
        stage_eval(opts.workers)
    print(json.dumps({"stage": opts.stage, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
