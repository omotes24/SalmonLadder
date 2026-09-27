"""Phase 3 baseline: LoCoOp (Miyai et al., NeurIPS 2023) with the authors' 16-shot ImageNet-1K checkpoints
(seeds 1-3, ViT-B/16, 16 context tokens, class token at the end), scored with MCM and GL-MCM exactly as the authors'
test_ood (logits / 100, softmax with T = 1; global: max over classes; local: max over patches and classes;
GL-MCM = global + local). The model code is the authors' clip_w_local (ext/LoCoOp), unchanged.
Class names: CLIP-style clean ImageNet names (the same list TINS uses); the CoOp classnames.txt is not shipped with the
repository. The checkpoint's fixed token buffers are ignored, as in the authors' load_model.

Usage: python scripts/p3_locoop.py --seed 1 --ids <npy/list file with sample ids> --paths <file with paths> --out F
       (every path passes the sealing guard unless --unseal <pre-registration sha256> matches the frozen file)
Output: npz with sample_id, mcm, glmcm (higher = more ID), top1 (argmax class of the global logits)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402

EXT = ROOT / "ext"
N_CTX = 16


class Images(Dataset):
    def __init__(self, paths, transform, check):
        self.paths, self.transform, self.check = list(paths), transform, check

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.check(self.paths[i])).convert("RGB")), i


def build(seed, device):
    sys.path.insert(0, str(EXT / "LoCoOp"))
    import clip_w_local
    from clip_w_local import clip as cw

    original = cw._download
    cw._download = lambda url, root=None: original(url, str(C.CLIP_WEIGHTS_DIR))
    try:
        model, preprocess = clip_w_local.load("ViT-B/16", device=device)
    finally:
        cw._download = original
    model.eval()
    from vins.splits import load_imagenet_classes

    _, _, clean = load_imagenet_classes()
    names = [n.replace("_", " ") for n in clean]
    prompts = [" ".join(["X"] * N_CTX) + " " + n + "." for n in names]
    tok = torch.cat([clip_w_local.tokenize(p) for p in prompts]).to(device)
    ck = torch.load(EXT / "locoop_ckpt" / f"seed{seed}" / "prompt_learner" / "model.pth.tar-50", map_location="cpu",
                    weights_only=False)
    assert ck["epoch"] == 50
    ctx = ck["state_dict"]["ctx"].to(device).type(model.dtype)
    with torch.no_grad():
        emb = model.token_embedding(tok).type(model.dtype)
        x = torch.cat([emb[:, :1], ctx.unsqueeze(0).expand(len(names), -1, -1), emb[:, 1 + N_CTX:]], dim=1)
        x = x + model.positional_embedding.type(model.dtype)
        x, _, _, _ = model.transformer(x.permute(1, 0, 2))
        x = model.ln_final(x.permute(1, 0, 2)).type(model.dtype)
        text = x[torch.arange(len(names)), tok.argmax(dim=-1)] @ model.text_projection
        text = text / text.norm(dim=-1, keepdim=True)
    return model, preprocess, text


@torch.no_grad()
def score(model, preprocess, text, paths, check, T=1.0, batch=100):
    loader = DataLoader(Images(paths, preprocess, check), batch_size=batch, num_workers=8, shuffle=False, pin_memory=True)
    mcm, glmcm, top1, order = [], [], [], []
    scale = model.logit_scale.exp()
    for img, idx in loader:
        g, loc = model.visual(img.cuda(non_blocking=True).type(model.dtype))
        g = g / g.norm(dim=-1, keepdim=True)
        loc = loc / loc.norm(dim=-1, keepdim=True)
        out = (scale * g @ text.t()).float() / 100.0
        out_l = (scale * loc @ text.t()).float() / 100.0
        sg = torch.softmax(out / T, dim=-1)
        sl = torch.softmax(out_l / T, dim=-1)
        m_g = sg.max(dim=1).values
        m_l = sl.amax(dim=(1, 2))
        mcm.append(m_g.cpu())
        glmcm.append((m_g + m_l).cpu())
        top1.append(out.argmax(dim=1).cpu())
        order.append(idx)
    order = torch.cat(order)
    assert torch.equal(order, torch.arange(len(paths)))
    return torch.cat(mcm).numpy(), torch.cat(glmcm).numpy(), torch.cat(top1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unseal", default=None, help="sha256 of the frozen Phase 3 pre-registration")
    opts = parser.parse_args()
    start = time.time()
    ids = [x for x in Path(opts.ids).read_text().splitlines() if x]
    paths = [x for x in Path(opts.paths).read_text().splitlines() if x]
    assert len(ids) == len(paths)
    from vins.features import guard_from_seal

    guard = guard_from_seal()
    if opts.unseal:
        prereg = C.WORK / "r5" / "phase3" / "prereg_phase3.json"
        digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
        assert digest == opts.unseal, "the Phase 3 pre-registration does not match; sealed images stay sealed"
        check = str
    else:
        check = guard.check
    model, preprocess, text = build(opts.seed, "cuda")
    mcm, glmcm, top1 = score(model, preprocess, text, paths, check)
    np.savez_compressed(opts.out, sample_id=np.array(ids), mcm=mcm, glmcm=glmcm, top1=top1)
    print(json.dumps({"n": len(ids), "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
