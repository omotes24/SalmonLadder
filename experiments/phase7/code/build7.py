"""Phase 7 dataset banks for the dataset-dependence study (D). No image is scored here.

Per dataset: <P7>/banks/<ds>/images.parquet (sample_id, path, cls, part), split<k>.parquet (sample_id, role, cls, cls_rank,
pos; role in sup / cal / id / ood) and split<k>.json (ID class names in label order, unknown classes, counts).
Class splits and shots are seeded with [SEED, dataset code, split]. Protocols are those of prereg_p7.json (D_datasets).
"""
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from p7common import BANKS, P4, SEED, dump, sha_file, utc

CUB = Path("/home/omote/reprise_controls_20260927/data/CUB_200_2011")
CIFAR = Path("/home/omote/datasets/cifar100/cifar-100-python")
PLACES = Path("/home/omote/datasets/places365_val256")
MOS_PLACES = Path("/home/omote/datasets/extra_ood/Places/images")
INR = Path("/home/omote/datasets/extra_ood/imagenet-r")
INSK = Path("/home/omote/datasets/extra_ood/imagenet-sketch/sketch")
CODE = {"cub": 1, "cifar100": 2, "places365": 3, "inr": 4, "insk": 5}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_SPLITS = 5


def cub():
    rd = lambda fn: [line.split(" ", 1) for line in (CUB / fn).read_text().splitlines() if line.strip()]
    paths = {int(i): p for i, p in rd("images.txt")}
    lab = {int(i): int(c) for i, c in rd("image_class_labels.txt")}
    tr = {int(i): int(t) for i, t in rd("train_test_split.txt")}
    names = {f"{int(i):03d}": n.split(".", 1)[1].replace("_", " ") for i, n in rd("classes.txt")}
    rows = [{"sample_id": f"cub/{paths[i]}", "path": str(CUB / "images" / paths[i]), "cls": f"{lab[i]:03d}",
             "part": "train" if tr[i] else "test"} for i in sorted(paths)]
    return pd.DataFrame(rows), names, dict(n_unknown=50, shot=("train",), idp=("test",), oodp=("test",))


def cifar100():
    out = BANKS / "cifar100" / "images"
    meta = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")
    names = {f"{i:03d}": n.replace("_", " ") for i, n in enumerate(meta["fine_label_names"])}
    rows = []
    for part in ("train", "test"):
        blob = pickle.load(open(CIFAR / part, "rb"), encoding="latin1")
        data = np.asarray(blob["data"], np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        (out / part).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(blob["fine_labels"]):
            p = out / part / f"{i:05d}.png"
            if not p.exists():
                Image.fromarray(data[i]).save(p)
            rows.append({"sample_id": f"cifar100/{part}/{i:05d}", "path": str(p), "cls": f"{int(c):03d}", "part": part})
    return pd.DataFrame(rows), names, dict(n_unknown=50, shot=("train",), idp=("test",), oodp=("test",))


def places365():
    from vins.sealing import load_sealed

    cats = {}
    for line in (PLACES / "categories_places365.txt").read_text().splitlines():
        if line.strip():
            name, idx = line.split()
            cats[int(idx)] = name[3:].replace("_", " ").replace("/", " ")
    val = pd.read_csv(PLACES / "places365_val.txt", sep=" ", header=None, names=["file", "label"])
    paths = [str(PLACES / "val_256" / f) for f in val.file]
    mos = sorted(str(p) for p in MOS_PLACES.iterdir() if p.suffix.lower() in EXTS)
    with mp.get_context("fork").Pool(12) as pool:
        h_val = pool.map(sha_file, paths, chunksize=256)
        h_mos = set(pool.map(sha_file, mos, chunksize=256))
    _, sealed_hashes, _ = load_sealed()
    drop = np.array([h in sealed_hashes or h in h_mos for h in h_val])
    rows = [{"sample_id": f"places365/val/{f}", "path": p, "cls": f"{int(l):03d}", "part": "val"}
            for f, p, l, d in zip(val.file, paths, val.label, drop) if not d]
    names = {f"{i:03d}": n for i, n in cats.items()}
    extra = {"n_val": len(val), "dropped_sealed_or_four_ood": int(drop.sum()), "n_four_ood_places": len(mos)}
    return pd.DataFrame(rows), names, dict(n_unknown=73, shot=("val",), idp=("val",), oodp=("val",), extra=extra)


def imagenet_names():
    pool = pd.read_parquet(P4 / "banks" / "imagenet_pool.parquet").drop_duplicates("wnid")
    return dict(zip(pool.wnid, pool.clean_name))


def folder(root, prefix):
    rows = []
    for c in sorted(p.name for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(p for p in (Path(root) / c).iterdir() if p.is_file() and p.suffix.lower() in EXTS):
            rows.append({"sample_id": f"{prefix}/{c}/{f.name}", "path": str(f), "cls": c, "part": "all"})
    return pd.DataFrame(rows)


def inr():
    im = folder(INR, "inr")
    nm = imagenet_names()
    return im, {c: nm[c] for c in im.cls.unique()}, dict(n_unknown=40, shot=("all",), idp=("all",), oodp=("all",), cap_id=100, cap_ood=100)


def insk():
    im = folder(INSK, "insk")
    nm = imagenet_names()
    fixed = {}
    for k in range(1, N_SPLITS + 1):
        sp = json.loads((P4 / "banks" / "U1" / f"split{k}.json").read_text())
        fixed[k] = {"id": list(sp["id_wnids"]), "unknown": list(sp["heldout"])}
    return im, {c: nm[c] for c in im.cls.unique()}, dict(fixed=fixed, shot=("all",), idp=("all",), oodp=("all",), cap_id=20, cap_ood=50)


def make_splits(ds, images, names, n_unknown=None, shot=(), idp=(), oodp=(), cap_id=None, cap_ood=None, fixed=None, extra=None):
    out = BANKS / ds
    out.mkdir(parents=True, exist_ok=True)
    if (out / "build_info.json").exists():
        print(ds, "already built; refusing to overwrite")
        return
    images = images.sort_values("sample_id").reset_index(drop=True)
    assert images.sample_id.is_unique
    images.to_parquet(out / "images.parquet", index=False)
    classes = sorted(images.cls.unique())
    by = {c: g for c, g in images.groupby("cls")}
    info = {"dataset": ds, "utc": utc(), "seed": [SEED, CODE[ds]], "n_images": len(images), "n_classes": len(classes), "splits": [], **(extra or {})}
    for k in range(1, N_SPLITS + 1):
        rng = np.random.default_rng([SEED, CODE[ds], k])
        if fixed:
            idc, unknown = fixed[k]["id"], sorted(fixed[k]["unknown"])
        else:
            perm = rng.permutation(len(classes))
            unknown = sorted(classes[i] for i in perm[:n_unknown])
            idc = sorted(classes[i] for i in perm[n_unknown:])
        rows = []
        for j, c in enumerate(idc):
            g = by[c]
            sp = g[g.part.isin(shot)]
            pick = rng.permutation(len(sp))[:16]
            assert len(pick) == 16, (ds, c, len(sp))
            ids = sp.sample_id.values[pick]
            rows += [{"sample_id": s, "role": "sup" if p < 12 else "cal", "cls": c, "cls_rank": j, "pos": p if p < 12 else p - 12}
                     for p, s in enumerate(ids)]
            rest = g[g.part.isin(idp) & ~g.sample_id.isin(set(ids))].sample_id.values
            if cap_id is not None and len(rest) > cap_id:
                rest = rest[np.sort(rng.permutation(len(rest))[:cap_id])]
            rows += [{"sample_id": s, "role": "id", "cls": c, "cls_rank": j, "pos": p} for p, s in enumerate(rest)]
        for c in unknown:
            g = by[c]
            o = g[g.part.isin(oodp)].sample_id.values
            if cap_ood is not None and len(o) > cap_ood:
                o = o[np.sort(rng.permutation(len(o))[:cap_ood])]
            rows += [{"sample_id": s, "role": "ood", "cls": c, "cls_rank": -1, "pos": p} for p, s in enumerate(o)]
        f = pd.DataFrame(rows)
        assert f.sample_id.is_unique
        f.to_parquet(out / f"split{k}.parquet", index=False)
        cnt = f.role.value_counts().to_dict()
        ood_per = f[f.role == "ood"].groupby("cls").size()
        meta = {"split": k, "names": [names[c] for c in idc], "id_classes": idc, "unknown": unknown,
                "unknown_names": [names[c] for c in unknown], "counts": {r: int(v) for r, v in cnt.items()},
                "ood_share": cnt.get("ood", 0) / (cnt.get("ood", 0) + cnt.get("id", 0)),
                "ood_per_class": {"min": int(ood_per.min()), "median": float(ood_per.median()), "max": int(ood_per.max())}}
        dump(out / f"split{k}.json", meta)
        info["splits"].append({**{k_: meta[k_] for k_ in ("split", "counts", "ood_share", "ood_per_class")},
                               "n_id_classes": len(idc), "n_unknown": len(unknown), "sha256": sha_file(out / f"split{k}.parquet")})
    dump(out / "build_info.json", info)
    print(json.dumps({"dataset": ds, "n_images": len(images), "splits": info["splits"][:2], **(extra or {})}), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:] or list(CODE):
        im, nm, kw = {"cub": cub, "cifar100": cifar100, "places365": places365, "inr": inr, "insk": insk}[ds]()
        make_splits(ds, im, nm, **kw)
