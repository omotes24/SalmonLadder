"""Encoding cost per view: parameters of the image encoder and milliseconds per image on one RTX 2080 Ti
(fp32, TF32 off, batch 64 [g/14: 32], synthetic input of the view's resolution, 3 warm-up + 10 timed batches).
Run on an idle GPU. Output: results/bench7.json"""
import json
import sys
import time

import torch

import extract7 as X
from p7common import RESULTS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
RES = {"dino224": 224, "dino256": 256, "clip_tins": 224, "hf_clipl": 224, "hf_sig2l": 256, "clip_rn50": 224}
out = {}
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["S14", "B14", "L14", "G14", "D3B", "D3L", "DINO1", "MAE", "CLIP", "CLIPL", "SIG2L"]
for n in names:
    enc, tkey, _, nparam = X.load(n, "cuda", RESULTS / "bench_cache")
    bs = 32 if n == "G14" else 64
    x = torch.randn(bs, 3, RES[tkey], RES[tkey], device="cuda")
    with torch.no_grad():
        for _ in range(3):
            X.pooled(enc(x))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            X.pooled(enc(x))
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / (10 * bs)
    out[n] = {"params_M": round(nparam / 1e6, 1), "ms_per_image": round(ms, 2), "input": RES[tkey], "batch": bs,
              "gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    print(n, out[n], flush=True)
    del enc
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
f = RESULTS / (sys.argv[2] if len(sys.argv) > 2 else "bench7.json")     # a second argument writes to another file
old = json.loads(f.read_text()) if f.exists() else {}
old.update(out)
f.write_text(json.dumps(old, indent=1) + "\n")
