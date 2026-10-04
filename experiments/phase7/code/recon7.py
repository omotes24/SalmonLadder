"""Phase 7 reconnaissance (read-only): schemas of existing results and local data; no scoring."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/omote/reprise_p4_20260928")
P5 = Path("/home/omote/reprise_p5_20261002")
G = Path("/home/omote/vins_gonogo_20260925")


def head(title):
    print(f"\n=== {title}", flush=True)


head("exp4 files")
fs = sorted(glob.glob(str(R / "results/exp4/*.parquet")))
print(len(fs), [os.path.basename(f) for f in fs[:3]], [os.path.basename(f) for f in fs if "_r0" in f])
d = pd.read_parquet(R / "results/exp4/s1_r0.parquet")
print("r0 columns", list(d.columns), len(d), d.family.unique().tolist(), d["query"].nunique())
f1 = [f for f in fs if "_r0" not in f and "auroc" not in f][0]
d = pd.read_parquet(f1)
print("class file columns", list(d.columns), len(d), d.cond.unique().tolist(), sorted(d.r.unique().tolist()), d["query"].nunique())
print(d.head(3).to_string())
a = pd.read_parquet(R / "results/exp4/auroc_table.parquet")
print("auroc_table", list(a.columns), len(a))
print(a.head(3).to_string())
print(open(R / "results/exp4_summary.md").read()[:3500])

head("eval score npz keys")
z = np.load(R / "results/eval/U1_s1_d0_seed123_scores.npz", allow_pickle=True)
print(z.files)
print({k: z[k].shape for k in z.files[:6]}, z["sample_id"][:2], int(z["is_ood"].sum()), len(z["is_ood"]))
e = pd.read_parquet(R / "results/eval/U1_s1_d0_seed123.parquet")
print(e[(e.base == "none") & (e.a == 1.0)][["family", "config", "role", "AUROC", "FPR95"]].to_string())

head("operating files")
print(sorted(os.path.basename(f) for f in glob.glob(str(R / "results/operating/*")))[:80])

head("pool")
pool = pd.read_parquet(R / "banks/imagenet_pool.parquet")
print(list(pool.columns), len(pool))
print(pool.role.value_counts().to_dict())
sp = json.loads((R / "banks/U1/split1.json").read_text())
print("split keys", list(sp.keys()), len(sp["heldout"]), len(sp["id_wnids"]))
held = set(sp["heldout"])
print("roles of held-out classes of split 1:", pool[pool.wnid.isin(held)].role.value_counts().to_dict())
ft = pd.read_parquet(R / "banks/feature_table.parquet")
print("feature_table", list(ft.columns), len(ft), ft.source.value_counts().to_dict())

head("U4")
p4 = pd.read_parquet(P5 / "banks/imagenet_pool.parquet")
print(list(p4.columns), len(p4), p4.role.value_counts().to_dict())
print(sorted(os.listdir(P5 / "banks/U4"))[:8], sorted(os.listdir(P5 / "results_final"))[:10])
print(sorted(os.listdir(P5 / "results_final/u4"))[:3], len(os.listdir(P5 / "results_final/u4")))
z = np.load(sorted(glob.glob(str(P5 / "results_final/u4/*.npz")))[0], allow_pickle=True)
print([k for k in z.files if not k.startswith("s::L14")][:60])

head("dev results")
for cfg in ("lock", "base"):
    fl = sorted(glob.glob(str(P5 / f"results/{cfg}/*.npz")))
    print(cfg, len(fl), os.path.basename(fl[0]) if fl else None)
    if fl:
        z = np.load(fl[0], allow_pickle=True)
        print("  keys(B14):", [k for k in z.files if k.startswith("s::B14::")][:40])
        print("  other:", [k for k in z.files if not k.startswith(("s::", "a::"))])

head("extra feats (ImageNet-R / Sketch / V2)")
xf = G / "extra_feats"
print(sorted(os.listdir(xf)) if xf.exists() else "missing")
if xf.exists():
    import torch
    for n in ("in_r.dino.pt", "in_sketch.dino.pt"):
        if (xf / n).exists():
            b = torch.load(xf / n, map_location="cpu")
            print(n, list(b.keys()), b["features"].shape, b["ids"][:2] if "ids" in b else None, b["paths"][:1])
            lab = np.asarray(b["labels"])
            cnt = np.bincount(lab[lab >= 0])
            cnt = cnt[cnt > 0]
            print("   classes", len(cnt), "per class min/median/max", cnt.min(), int(np.median(cnt)), cnt.max(), "total", len(lab))

head("CUB / CIFAR / Places")
cub = Path("/home/omote/reprise_controls_20260927/data/CUB_200_2011")
print("cub files", sorted(os.listdir(cub))[:12])
tt = [l.split() for l in (cub / "train_test_split.txt").read_text().splitlines()]
print("cub n", len(tt), "train", sum(int(t) for _, t in tt))
print("controls cub", sorted(os.listdir("/home/omote/reprise_controls_20260927/data/cub"))[:20])
print("controls cifar", sorted(os.listdir("/home/omote/reprise_controls_20260927/data/cifar"))[:20])
print("cifar raw", sorted(os.listdir("/home/omote/datasets/cifar100/cifar-100-python")))
pl = Path("/home/omote/datasets/places365_val256")
v = pd.read_csv(pl / "places365_val.txt", sep=" ", header=None, names=["file", "label"])
print("places val", len(v), v.label.nunique(), v.groupby("label").size().describe().to_dict())
print((pl / "categories_places365.txt").read_text().splitlines()[:5], len(os.listdir(pl / "val_256")))
print("places extra meta", (G / "r5/extra/places_info.json").exists())
if (G / "r5/extra/places_info.json").exists():
    print(json.loads((G / "r5/extra/places_info.json").read_text()))

head("models offline")
os.environ["HF_HUB_OFFLINE"] = "1"
import timm, torch
print("timm", timm.__version__, "torch", torch.__version__)
for name in ("vit_base_patch16_224.dino", "vit_base_patch16_224.mae", "resnet50_clip.openai", "vit_large_patch14_clip_224.openai",
             "vit_base_patch16_clip_224.openai"):
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_model_data_config(m)
        x = torch.zeros(2, 3, *cfg["input_size"][1:])
        with torch.no_grad():
            y = m(x)
        print(name, "ok dim", tuple(y.shape), "params(M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
              {k: cfg[k] for k in ("input_size", "interpolation", "mean", "std", "crop_pct")}, "pool", getattr(m, "global_pool", None))
    except Exception as ex:
        print(name, "FAILED", repr(ex)[:300])
hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
print("dinov2 hub dir", hub.exists())
print("done")
