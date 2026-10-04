"""Phase 3: LoCoOp (official 16-shot checkpoints, seeds 1-3) on OpenOOD v1.5 ImageNet-1K and Four-OOD.
MCM and GL-MCM per image (p3_locoop.py), then AUROC / FPR95 per OOD set against the ID set of the benchmark
(test_imagenet for OpenOOD, ImageNet-1K val for Four-OOD); static scores, so the stream order does not matter.
Requires --unseal <sha256 of the frozen pre-registration>. Output: <R5>/phase3/locoop_metrics.json (+ per-image npz)
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
PREFIX = {"openimageo": "openimage_o"}                                  # sample-id prefix of OpenImage-O


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseal", required=True)
    opts = parser.parse_args()
    digest = hashlib.sha256((P3 / "prereg_phase3.json").read_bytes()).hexdigest()
    if opts.unseal != digest:
        raise SystemExit("the unseal token does not match the frozen Phase 3 pre-registration")
    start = time.time()
    spec = importlib.util.spec_from_file_location("p3_locoop", ROOT / "scripts" / "p3_locoop.py")
    LC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LC)
    from vins.tins_dev import import_tins

    import_tins()
    from vins.metrics import measures as upstream

    b = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    ids = sorted({x for ds in OO for x in np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist()})
    sets = {"openood": (ids, b["paths"][16000:])}
    names = ["in_val"] + FOUR
    blobs = {n: torch.load(C.WORK / "extra_feats" / f"{n}.dino.pt", map_location="cpu") for n in names}
    sets["fourood"] = (sum((blobs[n]["ids"] for n in names), []), sum((list(blobs[n]["paths"]) for n in names), []))
    out = {}
    todo = [part for part in sets if not all((P3 / f"locoop_seed{s}_{part}.npz").exists() for s in (1, 2, 3))]
    if todo:
        model, preprocess, _ = LC.build(1, "cuda")                   # the image encoder is the same for every seed
        texts = {seed: LC.text_features(model, seed, "cuda") for seed in (1, 2, 3)}
    for part, (sid, paths) in sets.items():
        if part not in todo:
            continue
        res = LC.score_multi(model, preprocess, texts, [str(p) for p in paths], str)
        for seed, r in res.items():
            np.savez_compressed(P3 / f"locoop_seed{seed}_{part}.npz", sample_id=np.array(sid), **r)
    for seed in (1, 2, 3):
        for part in sets:
            z = np.load(P3 / f"locoop_seed{seed}_{part}.npz")
            s_id = np.array(z["sample_id"]).astype(str)
            prefix = np.array([x.rsplit("_", 1)[0] for x in s_id])
            id_mask = prefix == ("imagenet" if part == "openood" else "in_val")
            for score in ("mcm", "glmcm"):
                sc = z[score].astype(np.float64)
                for ds in (OO if part == "openood" else FOUR):
                    m = upstream(sc[id_mask], sc[prefix == PREFIX.get(ds, ds)])
                    out.setdefault(part, {}).setdefault(score, {}).setdefault(ds, []).append(
                        {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]})
    summary = {}
    for part, d in out.items():
        for score, per in d.items():
            for ds, vals in per.items():
                summary.setdefault(part, {}).setdefault(score, {})[ds] = {
                    k: float(np.mean([v[k] for v in vals])) for k in ("AUROC", "FPR95")}
            if part == "openood":
                for g, dss in (("near", OO[:2]), ("far", OO[2:])):
                    summary[part][score][g] = {k: float(np.mean([summary[part][score][x][k] for x in dss])) for k in ("AUROC", "FPR95")}
            else:
                summary[part][score]["avg"] = {k: float(np.mean([summary[part][score][x][k] for x in FOUR])) for k in ("AUROC", "FPR95")}
    (P3 / "locoop_metrics.json").write_text(json.dumps({"summary": summary, "per_seed": out,
                                                        "seconds": round(time.time() - start, 1)}, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
