"""Exp 2 resource columns, measured on one idle GPU for one confirmatory stream (U1 split 1, draw 0, order 123;
23,000 images, batch 256): encoder forward time (DINOv2 B/14, L/14, CLIP ViT-B/16; images preloaded, FP32, TF32
off), TINS (setup and stream, from cached CLIP features), frozen REPRISE and the best minimal configuration (both
views, static part, graph, memory, propagation), with peak allocated / reserved GPU memory per component.
Output: results/resources.json"""
import json
import time

import numpy as np
import pandas as pd
import torch

from bank_data import TINS_DIR, Features, shots_of, spec, stream_inputs
from common import BANKS, RESULTS, V5, utc
from engine import run_view
from static import static_view

BANK, SPLIT, DRAW, SEED = "U1", 1, 0, 123


def peak_reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def peak(base):
    torch.cuda.synchronize()
    return {"alloc_MiB": (torch.cuda.max_memory_allocated() - base) / 2 ** 20,
            "reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20}


@torch.no_grad()
def encoders(ids, n=512):
    from PIL import Image
    from extract import load_models
    from vins.features import dino_transform
    from vins.sealing import Guard, load_sealed
    table = pd.read_parquet(BANKS / "feature_table.parquet").set_index("sample_id")
    models, pre = load_models()
    guard = Guard(load_sealed()[0])
    td, tc = dino_transform(), pre
    xd, xc = [], []
    for s in ids[:n]:
        with Image.open(guard.check(table.loc[s, "path"])) as im:
            im = im.convert("RGB")
            xd.append(td(im))
            xc.append(tc(im))
    xd, xc = torch.stack(xd).cuda(), torch.stack(xc).cuda()
    out = {}
    for name in ("B14", "L14", "CLIP"):
        base = peak_reset()
        reps = []
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for lo in range(0, n, 256):
                if name == "CLIP":
                    m = models["CLIP"]
                    m.encode_image(xc[lo:lo + 256].to(next(m.parameters()).dtype))
                else:
                    models[name].forward_features(xd[lo:lo + 256])
            torch.cuda.synchronize()
            reps.append(time.perf_counter() - t0)
        out[name] = {"ms_per_image": 1000 * min(reps) / n, **peak(base)}
    del models
    torch.cuda.empty_cache()
    return out


def visual(F, inp, families, memory):
    res = {"static_s": 0.0, "graph_s": 0.0, "memory_s": 0.0, "lp_s": 0.0}
    base = peak_reset()
    for v in ("B14", "L14"):
        t0 = time.perf_counter()
        st = static_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"], inp["cand_c"], inp["cand_s"], V5["n0"], V5["m"])
        res["static_s"] += time.perf_counter() - t0
        _, diag = run_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"], inp["bidx"], st, V5["thresholds"],
                           ((10, 1.0),), (0.9,), "cuda", families, memory=memory, trace=True)
        for d in diag:
            for k in ("graph_s", "memory_s", "lp_s"):
                res[k] += d[k]
        res[f"last_batch_ms_{v}"] = 1000 * (diag[-1]["graph_s"] + diag[-1]["memory_s"] + diag[-1]["lp_s"])
    n = len(inp["ids"])
    res.update(peak(base))
    res["ms_per_image"] = 1000 * (res["static_s"] + res["graph_s"] + res["memory_s"] + res["lp_s"]) / n
    return res


def tins_timing(F):
    from vins.tins_dev import Recorder, build_order, get_logger, import_tins, load_clip, make_args, run_stream, setup_to_device
    names, shots, ide, ood = spec(BANK, SPLIT)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, DRAW, C)
    t = import_tins()
    args = make_args(t, TINS_DIR / "cache", f"p4_{BANK}_s{SPLIT}_d{DRAW}")
    t.setup_seed(args.seed)
    log = get_logger(TINS_DIR / "resources.log")
    net, _ = load_clip(t, args)
    device = next(net.parameters()).device
    base = peak_reset()
    t0 = time.perf_counter()
    clip_shots = torch.as_tensor(np.concatenate([F.get("CLIP", sup_ids).reshape(C, 12, -1), F.get("CLIP", cal_ids).reshape(C, 4, -1)], 1))
    protos = clip_shots.mean(1)
    protos = (protos / protos.norm(dim=-1, keepdim=True)).to(device)
    positive = t.encode_texts(net, [args.pos_prompt.format(x) for x in names], batch_size=args.text_batch_size, device=device, desc="pos").to(device)
    neg, _, words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=names, positive_features=positive,
                                                     class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, words, protos, device, log)
    setup = {"positive_features": positive.cpu(), "negative_features": neg.cpu(), "class_prototypes": protos.cpu(),
             "base_sim": (positive * protos).sum(dim=1).cpu(),
             "init_candidates": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in init.items()}}
    dev_setup = setup_to_device(setup)
    torch.cuda.synchronize()
    t_setup = time.perf_counter() - t0
    order = build_order(len(ide), len(ood), SEED)
    ids = [ide[i] if o == 0 else ood[i] for o, i in order]
    feats = torch.as_tensor(F.get("CLIP", ids))
    args.stream_seed = SEED
    t.setup_seed(args.seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_stream(t, args, net, dev_setup, feats, hook=Recorder())
    torch.cuda.synchronize()
    t_stream = time.perf_counter() - t0
    return {"setup_s": t_setup, "stream_s": t_stream, "ms_per_image": 1000 * t_stream / len(ids), **peak(base),
            "note": "negative bank loaded from the per-split cache if present (setup time then excludes its construction)"}


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    F = Features()
    inp = stream_inputs(F, BANK, SPLIT, DRAW, SEED)
    out = {"utc": utc(), "stream": f"{BANK}_s{SPLIT}_d{DRAW}_seed{SEED}", "n": len(inp["ids"]),
           "gpu": torch.cuda.get_device_name(0)}
    out["REPRISE_frozen"] = visual(F, inp, ("cdf_L0",), True)
    out["minimal_z"] = visual(F, inp, ("z",), False)
    out["encoders"] = encoders(inp["ids"])
    out["TINS"] = tins_timing(F)
    (RESULTS / "resources.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
