"""Merge the shards of extract5.py into <P5>/features/<set>.<name>.pt (sample_id, features)."""
import glob
import sys
from pathlib import Path

import torch

F = Path("/home/omote/reprise_p5_20261002/features")
for which in sys.argv[1:] or ("shots", "dev1", "dev2"):
    for name in ("B14pm", "L14pm", "G14", "G14pm", "SIG2L", "CLIPL"):
        parts = sorted(p for p in glob.glob(str(F / f"{which}.{name}.shard*of*.pt")) if "limit" not in p)
        if not parts:
            continue
        n = int(parts[0].split("of")[-1].split(".")[0])
        if len(parts) != n:
            print(which, name, "incomplete", len(parts), "of", n)
            continue
        ids, feats = [], []
        for p in parts:
            b = torch.load(p, map_location="cpu")
            ids += b["sample_id"]
            feats.append(b["features"])
        assert len(set(ids)) == len(ids)
        torch.save({"sample_id": ids, "features": torch.cat(feats), "meta": {"set": which, "name": name, "shards": n}}, F / f"{which}.{name}.pt")
        print(which, name, len(ids), tuple(torch.cat(feats).shape))
