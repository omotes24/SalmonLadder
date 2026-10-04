"""Exp 2 (original methods): the official AdaNeg (NeurIPS'24) and TANL (CVPR'26) postprocessors of OpenOOD-VLM
(commit in vlm_tta/SOURCE_COMMIT_OpenOOD-VLM), run unmodified on the confirmatory streams.

Text side: the official FixedCLIP_NegOODPrompt constructor (NegLabel WordNet corpus from the NegLabel repository,
10,000 negative labels, 'nice' prompt, text_center, fp16 CLIP ViT-B/16 from the sha256-verified local checkpoint),
with get_class_names patched to the stream's ID names (the names TINS and REPRISE receive).
Image side: the cached CLIP ViT-B/16 features of the confirmatory evaluation (L2-normalised, fp16 as the official
encoder output), fed through a network stub whose forward returns (features, text_features^T, logit_scale).
Official hyper-parameters: AdaNeg scripts/ood/adaneg/imagenet.sh (thres 0.5, gap 0.5, memleng 10, lambda 0.1,
beta 5.5 -> int 5, 5 groups, random permutation, combine; samada True = full AdaNeg, False = task-adaptive only);
TANL scripts/ood/TANL/official.sh (beta 1000, gamma 1, alpha 0, memleng 300, thres 0.5, gap 0.5, sum).
NegLabel = AdaNeg's vanilla score (in_score vanillaonly), as the repository notes.
Per (bank, split): one network; per order seed: one stream, batches of 256, memory reset per stream.
Output: results/vlm_tta/<bank>_s<split>_seed<seed>.npz (scores per method, per-batch seconds, peak GPU memory).
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VT = HERE / "vlm_tta"
sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
sys.path.insert(0, str(VT))

from bank_data import Features, spec  # noqa: E402
from common import RESULTS, dump, utc  # noqa: E402
from vins import config as C  # noqa: E402
from vins.tins_dev import build_order  # noqa: E402

OUT = RESULTS / "vlm_tta"
BATCH = 256


class A(dict):
    __getattr__ = dict.__getitem__


ADANEG = dict(tau=1.0, beta=5.5, in_score="combine", memleng=10, lambdaval=0.1, thres=0.5, gap=0.5,
              group_num=5, random_permute=True)
TANL = dict(tau=1.0, beta=1000, in_score="sum", alpha=0.0, gamma=1, group_num=5, group_size=1000,
            random_permute=False, thres=0.5, samada=True, gap=0.5, cluster_num=0, cossim=False, memleng=300)
METHODS = {"AdaNeg": ("ada", dict(ADANEG, samada=True)),
           "AdaNeg_TA": ("ada", dict(ADANEG, samada=False)),
           "NegLabel": ("ada", dict(ADANEG, samada=False, in_score="vanillaonly", thres=1.0)),
           "TANL": ("tanl", TANL)}


def import_official():
    import openood.networks.clip.clip as vclip
    try:
        import clip as pclip
    except ImportError:
        sys.modules["clip"] = vclip
        pclip = vclip
    for mod in {id(vclip): vclip, id(pclip): pclip}.values():
        orig = mod._download
        mod._download = (lambda o: (lambda url, root=None: o(url, str(C.CLIP_WEIGHTS_DIR))))(orig)
    from openood.networks import clip_fixed_ood_prompt as cfp
    from openood.postprocessors.adaneg_chunked import AdaNegChunked      # official body; chunked scoring (11 GB GPUs)
    from openood.postprocessors.activated_neg_postprocessor import ActivatedNegPostprocessor
    return cfp, AdaNegChunked, ActivatedNegPostprocessor


class StreamNet:
    """Holds the official network's text side; forward returns cached image features (no image is re-encoded)."""

    def __init__(self, net):
        for k in ("text_features", "text_features_unselected", "text_features_all", "noise_image_features",
                  "logit_scale", "n_cls", "n_output"):
            setattr(self, k, getattr(net, k))

    def eval(self):
        return self

    def __call__(self, x, return_feat=False):
        assert return_feat
        return x, self.text_features.transpose(0, 1), self.logit_scale


def text_path(bank, split):
    return OUT / f"text_{bank}_s{split}.pt"


class Cached:
    pass


def load_or_build(cfp, bank, split, names):
    """Text side is deterministic given the ID names; cached so that streams can run after features exist."""
    p = text_path(bank, split)
    if p.exists():
        d = torch.load(p, map_location="cuda")
        assert d["names"] == list(names)
        s = Cached()
        for k, v in d.items():
            setattr(s, k, v)
        s.__class__ = StreamNet
        return s, d["setup_s"]
    t0 = time.time()
    s = build_net(cfp, bank, split, names)
    t = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    d = {k: getattr(s, k) for k in ("text_features", "text_features_unselected", "text_features_all",
                                     "noise_image_features", "logit_scale", "n_cls", "n_output")}
    d.update(names=list(names), setup_s=t, utc=utc())
    tmp = str(p) + f".{os.getpid()}"
    torch.save(d, tmp)
    os.replace(tmp, p)
    return s, t


def build_net(cfp, bank, split, names):
    key = f"p4{bank}s{split}"
    orig = cfp.get_class_names
    cfp.get_class_names = lambda d: list(names) if d == key else orig(d)
    cwd = os.getcwd()
    os.chdir(VT)                                   # official code reads ./data/txtfiles, writes ./data/txtfiles_output
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        net = cfp.FixedCLIP_NegOODPrompt(A(backbone=A(name="ViT-B/16", dataset=key, text_prompt="nice",
                                                      text_center=True, ood_number=10000)))
    finally:
        os.chdir(cwd)
        cfp.get_class_names = orig
    s = StreamNet(net)
    del net
    torch.cuda.empty_cache()
    return s


def make_pp(kind, args, classes):
    cfg = A(postprocessor=A(postprocessor_args=A(args), postprocessor_sweep=A(tau_list=[1.0])))
    return classes[kind](cfg)


def run_stream(snet, feats, kind, args, classes, seed):
    torch.manual_seed(seed)
    pp = make_pp(kind, args, classes)
    pp.reset_memory()
    dev = snet.text_features.device
    scores, secs = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    for lo in range(0, len(feats), BATCH):
        x = torch.as_tensor(feats[lo:lo + BATCH], device=dev)
        x = (x / x.norm(dim=-1, keepdim=True)).to(snet.text_features.dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, conf = pp.postprocess(snet, x)
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        scores.append(conf.float().cpu().numpy())
    peak = {"alloc_MiB": (torch.cuda.max_memory_allocated() - base_alloc) / 2 ** 20,
            "reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20}
    return np.concatenate(scores).astype(np.float64), np.array(secs), peak


def job(F, classes, cfp, bank, split, seeds):
    todo = [s for s in seeds if not (OUT / f"{bank}_s{split}_seed{s}.npz").exists()]
    if not todo:
        return
    names, _, ide, ood = spec(bank, split)
    snet, t_text = load_or_build(cfp, bank, split, names)
    if F is None:
        print(json.dumps({"text_ready": f"{bank}_s{split}", "setup_s": round(t_text, 1), "n_neg": int(snet.text_features.shape[1] - snet.n_cls),
                          "n_all": int(snet.text_features_all.shape[1]), "utc": utc()}), flush=True)
        return
    for seed in todo:
        order = build_order(len(ide), len(ood), seed)
        ids = [ide[i] if o == 0 else ood[i] for o, i in order]
        flag = np.array([o for o, _ in order], bool)
        feats = F.get("CLIP", ids)
        res, meta = {}, {"bank": bank, "split": split, "seed": seed, "n": len(ids), "text_setup_s": t_text,
                         "n_neg": int(snet.text_features.shape[1] - snet.n_cls), "n_all": int(snet.text_features_all.shape[1]),
                         "dtype": str(snet.text_features.dtype), "utc": utc(), "methods": {}}
        for m, (kind, args) in METHODS.items():
            sc, secs, peak = run_stream(snet, feats, kind, args, classes, seed)
            assert np.isfinite(sc).all(), m
            res[m] = sc
            res[f"{m}__secs"] = secs
            meta["methods"][m] = {"ms_per_image": 1000 * secs.sum() / len(ids), "s_per_batch_median": float(np.median(secs)), **peak}
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = OUT / f"{bank}_s{split}_seed{seed}.{os.getpid()}.npz"
        np.savez_compressed(tmp, sample_id=np.array(ids), is_ood=flag, meta=json.dumps(meta), **res)
        os.replace(tmp, OUT / f"{bank}_s{split}_seed{seed}.npz")
        print(json.dumps({k: v for k, v in meta.items() if k != "methods"} | {"ms": {m: round(v["ms_per_image"], 3) for m, v in meta["methods"].items()}}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="U1:1,U1:2,U1:3,U1:4,U1:5,U2:0,U3:0")
    ap.add_argument("--seeds", default="123,124,125")
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args()
    cfp, Ada, Tanl = import_official()
    classes = {"ada": Ada, "tanl": Tanl}
    F = None if a.text_only else Features()
    seeds = [int(s) for s in a.seeds.split(",")]
    for j in a.jobs.split(","):
        b, s = j.split(":")
        job(F, classes, cfp, b, int(s), seeds)
    print("VLM_TTA_DONE", flush=True)
