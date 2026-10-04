"""Phase 6: fetch the two DINOv3 checkpoints (timm hub, approved by the user) and check the feature read-out.
No image is opened here (random inputs only)."""
import os
import time
from pathlib import Path

import timm
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
NAMES = {"B": "vit_base_patch16_dinov3.lvd1689m", "L": "vit_large_patch16_dinov3.lvd1689m"}
print("timm", timm.__version__, "torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)
for tag, name in NAMES.items():
    t0 = time.time()
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().cuda()
    cfg = m.pretrained_cfg
    print(tag, name, type(m).__name__, "params(M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "num_features", m.num_features,
          "prefix", getattr(m, "num_prefix_tokens", None), "global_pool", getattr(m, "global_pool", None),
          "dynamic", getattr(m, "dynamic_img_size", None), "load s", round(time.time() - t0, 1), flush=True)
    print(" cfg", {k: cfg.get(k) for k in ("hf_hub_id", "input_size", "crop_pct", "crop_mode", "interpolation", "mean", "std", "license", "num_classes")}, flush=True)
    torch.manual_seed(0)
    for s in (256, 224):
        x = torch.randn(4, 3, s, s).cuda()
        with torch.no_grad():
            f = m.forward_features(x)
            h = m.pool(f, pool_type="token")          # timm's default pooling of these models is the patch mean
            o = m(x)
            f2 = m.forward_features(x)
            f1 = m.forward_features(x[:1])
        print(" ", s, tuple(f.shape), "cls-vs-token-pool", float((f[:, 0] - h).abs().max()), "default-out-is-patch-mean", float((f[:, m.num_prefix_tokens:].mean(1) - o).abs().max()),
              "repeat", float((f - f2).abs().max()), "batch-invariance", float((f1[0, 0] - f[0, 0]).abs().max()),
              "cls norm", round(float(f[:, 0].norm(dim=-1).mean()), 3), flush=True)
    for s, bs in ((256, 64), (224, 64)):
        x = torch.randn(bs, 3, s, s).cuda()
        with torch.no_grad():
            m.forward_features(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                m.forward_features(x)
            torch.cuda.synchronize()
        print("   throughput", s, round(3 * bs / (time.time() - t0), 1), "img/s; peak GB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), flush=True)
    del m
    torch.cuda.empty_cache()
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
for name in NAMES.values():
    d = hub / ("models--timm--" + name)
    for f in sorted(d.glob("snapshots/*/*")):
        print("file", name, f.name, os.path.getsize(f), "blob", os.path.realpath(f).split("/")[-1], flush=True)
print("PROBE_DONE", flush=True)
