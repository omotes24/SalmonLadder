"""Amendment 03: fetch the three approved checkpoints (the only place of Phase 7 where a download is allowed) and check
the feature read-outs. No image is opened here (random inputs only).
  D3S   timm vit_small_patch16_dinov3.lvd1689m        (Hugging Face hub, timm organisation)
  D3SP  timm vit_small_plus_patch16_dinov3.lvd1689m   (Hugging Face hub, timm organisation)
  RN50  OpenAI CLIP RN50.pt                            (openaipublic.azureedge.net, sha256 verified by the loader)
Output: results/weights_p7b.json (file sizes, sha256, parameters, read-out checks)."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if "--offline" not in sys.argv:                      # must be set before huggingface_hub is imported
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

import timm  # noqa: E402
import torch  # noqa: E402

import extract7 as X  # noqa: E402
from p7common import RESULTS, utc  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ALLOWED = {"D3S": "vit_small_patch16_dinov3.lvd1689m", "D3SP": "vit_small_plus_patch16_dinov3.lvd1689m"}
assert all(X.TIMM_CLS0[k] == v for k, v in ALLOWED.items())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = {"utc": utc(), "timm": timm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "offline": "--offline" in sys.argv, "models": {}}
print(json.dumps({k: out[k] for k in ("utc", "timm", "torch", "gpu", "offline")}), flush=True)
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
for tag, name in ALLOWED.items():
    t0 = time.time()
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().cuda()
    cfg = m.pretrained_cfg
    info = {"timm": name, "class": type(m).__name__, "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "num_features": m.num_features,
            "prefix_tokens": getattr(m, "num_prefix_tokens", None), "global_pool": getattr(m, "global_pool", None), "load_seconds": round(time.time() - t0, 1),
            "cfg": {k: cfg.get(k) for k in ("hf_hub_id", "input_size", "crop_pct", "interpolation", "mean", "std", "license")}}
    torch.manual_seed(0)
    chk = {}
    for s in (256, 224):
        x = torch.randn(4, 3, s, s).cuda()
        with torch.no_grad():
            f = m.forward_features(x)
            h = m.pool(f, pool_type="token")            # timm's default pooling of these models is the patch mean
            o = m(x)
            f2 = m.forward_features(x)
            f1 = m.forward_features(x[:1])
        chk[str(s)] = {"tokens": list(f.shape), "cls_vs_token_pool": float((f[:, 0] - h).abs().max()),
                       "default_out_vs_patch_mean": float((f[:, m.num_prefix_tokens:].mean(1) - o).abs().max()),
                       "repeat": float((f - f2).abs().max()), "batch_invariance": float((f1[0, 0] - f[0, 0]).abs().max()),
                       "cls_norm": round(float(f[:, 0].norm(dim=-1).mean()), 3)}
    info["checks"] = chk
    # the read-out used by extract7 (CLS after the final norm) on the same input
    enc, tkey, _, n = X.load(tag, "cuda", RESULTS / "bench_cache")
    x = torch.randn(4, 3, 256, 256).cuda()
    with torch.no_grad():
        info["extract7_vs_cls"] = float((X.pooled(enc(x)) - m.forward_features(x)[:, 0]).abs().max())
    info["extract7_transform"] = tkey
    files = []
    for fpath in sorted((hub / ("models--timm--" + name)).glob("snapshots/*/*")):
        real = os.path.realpath(fpath)
        files.append({"file": fpath.name, "bytes": os.path.getsize(real), "sha256": sha(real)})
    info["files"] = files
    out["models"][tag] = info
    print(tag, json.dumps(info), flush=True)
    del m, enc
    torch.cuda.empty_cache()

# CLIP RN50 through the vendored OpenAI CLIP package (downloads RN50.pt into P7/weights on the first call)
arch, digest = X.OPENAI["RN50"]
t0 = time.time()
enc, tkey, pre, n = X.load("RN50", "cuda", RESULTS / "bench_cache")
f = X.WEIGHTS / "RN50.pt"
x = torch.randn(4, 3, 224, 224).cuda()
with torch.no_grad():
    y, y2, y1 = X.pooled(enc(x)), X.pooled(enc(x)), X.pooled(enc(x[:1]))
info = {"arch": arch, "params_M_image_tower": round(n / 1e6, 2), "out_dim": int(y.shape[1]), "dtype": str(y.dtype), "transform": tkey, "preprocess": str(pre).replace("\n", " "),
        "repeat": float((y.float() - y2.float()).abs().max()), "batch_invariance": float((y1[0].float() - y[0].float()).abs().max()),
        "finite": bool(torch.isfinite(y).all()), "load_seconds": round(time.time() - t0, 1),
        "files": [{"file": f.name, "bytes": os.path.getsize(f), "sha256": sha(f)}]}
assert info["files"][0]["sha256"] == digest, "RN50.pt: sha256 differs from the published one"
out["models"]["RN50"] = info
print("RN50", json.dumps(info), flush=True)
(RESULTS / "weights_p7b.json").write_text(json.dumps(out, indent=1) + "\n")
print("PROBE7B_DONE", flush=True)
