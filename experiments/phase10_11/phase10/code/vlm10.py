"""Phase 10: the official OpenOOD-VLM post-processors (TANL, AdaNeg, AdaNeg task-adaptive only, NegLabel) on the
public-benchmark streams, as base detectors and as measured baselines. The code of Phase 4 (vlm_tta.py: official
bodies vendored in reprise_p4_20260928/code/vlm_tta, official hyper-parameters, chunked AdaNeg scoring for 11 GB GPUs)
is used unchanged; only the ID names differ: the official ImageNet-1K names (dataset 'imagenet' of OpenOOD-VLM),
i.e. the official protocol of these methods on OpenOOD ImageNet-1K.
Image side: the cached CLIP ViT-B/16 features of the Phase 3 evaluation (the features TINS and MCM receive).
Stream side: the Phase 3 stream files (sample order, batches of 256); the memory of every method is reset per stream.
  --part openood|fourood  --worker w --nworkers n   |   --text-only (builds and caches the text side once)
Output: P10/results/vlm/<part>/<stream>.npz (score per method, sample_id, is_ood, meta)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import p10common as P

sys.path.insert(0, str(P.P4 / "code"))
import vlm_tta as VT  # noqa: E402  (official imports, StreamNet, run_stream, METHODS)

OUT = P.RESULTS / "vlm"
TEXT = OUT / "text_imagenet.pt"
METHODS = ["TANL", "AdaNeg", "AdaNeg_TA", "NegLabel"]


def build_text(cfp):
    """Official FixedCLIP_NegOODPrompt for dataset 'imagenet' (official 1,000 names, 10,000 negatives, 'nice' prompt,
    text_center); the official code caches its WordNet selection under vlm_tta/data/txtfiles_output (cwd = vlm_tta)."""
    if TEXT.exists():
        d = torch.load(TEXT, map_location="cuda")
        s = VT.Cached()
        for k, v in d.items():
            setattr(s, k, v)
        s.__class__ = VT.StreamNet
        return s, d["setup_s"]
    t0 = time.time()
    cwd = os.getcwd()
    os.chdir(VT.VT)
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        net = cfp.FixedCLIP_NegOODPrompt(VT.A(backbone=VT.A(name="ViT-B/16", dataset="imagenet", text_prompt="nice",
                                                            text_center=True, ood_number=10000)))
    finally:
        os.chdir(cwd)
    s = VT.StreamNet(net)
    del net
    torch.cuda.empty_cache()
    t = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    d = {k: getattr(s, k) for k in ("text_features", "text_features_unselected", "text_features_all",
                                     "noise_image_features", "logit_scale", "n_cls", "n_output")}
    d.update(dataset="imagenet", setup_s=t, utc=P.utc())
    tmp = str(TEXT) + f".{os.getpid()}"
    torch.save(d, tmp)
    os.replace(tmp, TEXT)
    return s, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args()
    cfp, Ada, Tanl = VT.import_official()
    classes = {"ada": Ada, "tanl": Tanl}
    snet, t_text = build_text(cfp)
    print(json.dumps({"text_ready": "imagenet", "setup_s": round(t_text, 1), "n_cls": int(snet.n_cls),
                      "n_neg": int(snet.text_features.shape[1] - snet.n_cls), "n_all": int(snet.text_features_all.shape[1]),
                      "utc": P.utc()}), flush=True)
    if a.text_only:
        return
    assert a.part
    out_dir = OUT / a.part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(a.part)[a.worker::a.nworkers]
    todo = [(ds, s, p) for (ds, s, p) in tasks if not (out_dir / f"{ds}_seed{s}.npz").exists()]
    if not todo:
        return
    D = P.load_p3(a.part)
    clip = D["ev"]["CLIP"]
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        rows = np.array([D["row"][x] for x in ids]) - 4000
        feats = np.asarray(clip[rows], np.float32)
        res, meta = {}, {"part": a.part, "stream": name, "seed": int(seed), "n": int(len(ids)), "dataset": "imagenet",
                         "n_neg": int(snet.text_features.shape[1] - snet.n_cls), "utc": P.utc(), "methods": {}}
        for m in METHODS:
            kind, args = VT.METHODS[m]
            sc, secs, peak = VT.run_stream(snet, feats, kind, args, classes, int(seed))
            assert np.isfinite(sc).all(), m
            res[m] = sc
            res[f"{m}__secs"] = secs
            meta["methods"][m] = {"ms_per_image": 1000 * secs.sum() / len(ids), **peak}
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, sample_id=np.array(ids), is_ood=flag, meta=json.dumps(meta), **res)
        os.replace(tmp, out_dir / f"{name}.npz")
        print(json.dumps({k: v for k, v in meta.items() if k != "methods"} |
                         {"ms": {m: round(v["ms_per_image"], 3) for m, v in meta["methods"].items()}}), flush=True)
    print("VLM10_DONE", flush=True)


if __name__ == "__main__":
    main()
