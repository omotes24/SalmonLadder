"""CLIP ViT-B/16 text features of the ID class names (TINS positive prompt, pinned loader) for the dataset banks:
<P7>/banks/<ds>/pos_s<k>.npy (C x 512, L2-normalised). Only ID names reach the detector."""
import json
import os
import sys

import numpy as np

from p7common import BANKS
from vins.tins_dev import import_tins, load_clip, make_args

cwd = os.getcwd()
t = import_tins()
args = make_args(t, BANKS / "clip_cache", "p7_text")
net, _ = load_clip(t, args)
os.chdir(cwd)
device = next(net.parameters()).device
print("prompt:", args.pos_prompt)
for ds in sys.argv[1:]:
    for k in range(1, 6):
        out = BANKS / ds / f"pos_s{k}.npy"
        if out.exists():
            continue
        names = json.loads((BANKS / ds / f"split{k}.json").read_text())["names"]
        pos = t.encode_texts(net, [args.pos_prompt.format(x) for x in names], batch_size=args.text_batch_size, device=device, desc="pos")
        pf = (pos / pos.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        tmp = str(out) + f".{os.getpid()}.npy"
        np.save(tmp, pf)
        os.replace(tmp, out)
        print(ds, k, pf.shape, flush=True)
print("TEXT_OK")
