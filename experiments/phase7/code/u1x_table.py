"""Item list for the backbone study (B): shot, ID-evaluation and OOD-evaluation images of the Phase 4 pool, plus
ImageNet-O. The long-stream images and WILD are not needed. Output: <P7>/features/u1x/table.parquet"""
import pandas as pd

from p7common import FEATS, P4

pool = pd.read_parquet(P4 / "banks" / "imagenet_pool.parquet")
sel = pool[pool.role.isin(["shot", "ideval", "oodeval"])][["sample_id", "path"]]
o = pd.read_parquet(P4 / "banks" / "U2_imagenet_o.parquet")[["sample_id", "path"]]
t = pd.concat([sel, o]).drop_duplicates("sample_id").reset_index(drop=True)
out = FEATS / "u1x"
out.mkdir(parents=True, exist_ok=True)
if not (out / "table.parquet").exists():
    t.to_parquet(out / "table.parquet", index=False)
print("u1x table", len(t), "pool rows", len(sel), "imagenet-o", len(o))
