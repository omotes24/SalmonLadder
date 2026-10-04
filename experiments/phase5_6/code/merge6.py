"""Phase 6: merge the shards of extract6.py into <P5>/features/<set>.<name>.pt and check them against the item lists."""
import glob
import sys
from pathlib import Path

import torch

from extract6 import NAMES, items

F = Path("/home/omote/reprise_p5_20261002/features")
ok = True
for which in sys.argv[1:] or ("shots", "dev1", "dev2"):
    want, _ = items(which)
    for name in NAMES:
        parts = sorted(p for p in glob.glob(str(F / f"{which}.{name}.shard*of*.pt")) if "limit" not in p)
        if not parts:
            print(which, name, "no shards")
            ok = False
            continue
        n = int(parts[0].split("of")[-1].split(".")[0])
        if len(parts) != n:
            print(which, name, "incomplete", len(parts), "of", n)
            ok = False
            continue
        ids, feats = [], []
        for p in parts:
            b = torch.load(p, map_location="cpu")
            ids += b["sample_id"]
            feats.append(b["features"])
        f = torch.cat(feats)
        assert len(ids) == len(f) == len(want), (which, name, len(ids), len(f), len(want))
        first = {}
        for i, s in enumerate(ids):
            first.setdefault(s, i)
        keep = sorted(first.values())
        if len(keep) != len(ids):                      # an image listed in more than one draw: keep one row per sample_id
            print(which, name, "repeated sample_ids:", len(ids) - len(keep))
            ids, f = [ids[i] for i in keep], f[keep]
        assert set(ids) == set(want), (which, name)
        assert torch.isfinite(f).all() and torch.allclose(f.norm(dim=1), torch.ones(len(f)), atol=1e-4)
        torch.save({"sample_id": ids, "features": f, "meta": {"set": which, "name": name, "shards": n}}, F / f"{which}.{name}.pt")
        print(which, name, len(ids), tuple(f.shape), flush=True)
print("MERGE_OK" if ok else "MERGE_INCOMPLETE")
