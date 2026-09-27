"""R5 Phase 2 / M3 prerequisite: scene-type far OOD for the dev streams from Places365-Standard val, disjoint from the
Four-OOD Places subset (MOS) and from ImageNet-1K concepts. No test data; no label of the dev streams is used.

Classes = the 365 Places365 categories minus
  (a) the MOS Places classes (parsed from the Four-OOD file names "<letter>_<class>_<index>.jpg"),
  (b) categories whose first name segment or full name equals an ImageNet-1K class name or WordNet lemma
      (lower case, "_" and "/" as spaces; e.g. library, boathouse, valley, volcano, seashore, church).
The rest is split in two halves with a fixed seed: half 0 -> dev1, half 1 -> dev2. Per class PER_CLASS val images
(seeded choice). Images whose sha256 is sealed or equals a Four-OOD Places image are dropped.
Features: CLIP ViT-B/16 (TINS loader and preprocessing), DINOv2 B/14 and L/14 (standard eval preprocessing),
L2-normalised, the same code paths as the dev features.
Output: <R5>/extra/places_meta.parquet, places.clip.pt, places.dino.pt, places.dino_vitl14.pt, places_info.json
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import dino_encode_fn, dino_transform, encode, guard_from_seal, load_dino, save_features  # noqa: E402
from vins.r5 import R5  # noqa: E402
from vins.sealing import load_sealed  # noqa: E402
from vins.splits import load_imagenet_classes, sha256_file, wordnet_info  # noqa: E402

P365 = Path("/home/omote/datasets/places365_val256")
MOS = Path.home() / "datasets" / "extra_ood" / "Places" / "images"
PER_CLASS = 20
OUT = R5 / "extra"


def norm(s):
    return re.sub(r"\s+", " ", s.replace("_", " ").replace("/", " ").replace("-", " ").lower()).strip()


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "places_meta.parquet").exists():
        raise SystemExit("already built")
    cats = {}
    for line in (P365 / "categories_places365.txt").read_text().splitlines():
        name, idx = line.split()
        cats[int(idx)] = name[3:]                                # "/a/airfield" -> "airfield", "church/outdoor"
    mos_files = sorted(p.name for p in MOS.iterdir())
    mos_cls = {re.sub(r"_\d+\.jpg$", "", f)[2:] for f in mos_files}
    wnids, raw, clean = load_imagenet_classes()
    _, lemmas, _ = wordnet_info(wnids)
    in_names = {norm(x) for x in raw + clean} | {norm(le) for w in wnids for le in lemmas[w]}
    keep, dropped = [], {"mos": [], "imagenet": []}
    for idx, name in sorted(cats.items()):
        full = norm(name)
        first = norm(name.split("/")[0])
        if name.split("/")[0] in mos_cls or name.replace("/", "_") in mos_cls:
            dropped["mos"].append(name)
        elif first in in_names or full in in_names:
            dropped["imagenet"].append(name)
        else:
            keep.append(idx)
    rng = np.random.default_rng([20260927, 41])
    perm = rng.permutation(len(keep))
    half = {keep[i]: (0 if j < len(keep) // 2 else 1) for j, i in enumerate(perm)}
    val = pd.read_csv(P365 / "places365_val.txt", sep=" ", header=None, names=["file", "label"])
    _, sealed_hashes, _ = load_sealed()
    mos_hashes = {sha256_file(MOS / f) for f in mos_files}
    rows = []
    for idx in keep:
        files = np.sort(val.file.values[val.label.values == idx])
        pick = np.random.default_rng([20260927, 42, idx]).permutation(len(files))
        n = 0
        for j in pick:
            path = P365 / "val_256" / files[j]
            digest = sha256_file(path)
            if digest in sealed_hashes or digest in mos_hashes:
                continue
            rows.append({"sample_id": f"places365/val/{files[j]}", "path": str(path), "cls": cats[idx],
                         "label": idx, "dev": "dev1" if half[idx] == 0 else "dev2", "sha256": digest})
            n += 1
            if n == PER_CLASS:
                break
    meta = pd.DataFrame(rows)
    assert meta.sha256.is_unique
    ids, paths = meta.sample_id.tolist(), meta.path.tolist()
    guard = guard_from_seal()
    from vins.tins_dev import import_tins, load_clip, make_args

    t = import_tins()
    args = make_args(t, OUT / "clip_cache", "vins_r5_places")
    net, preprocess = load_clip(t, args)
    save_features(OUT / "places.clip.pt", ids, encode(paths, preprocess, net.encode_image, guard, 256, 8, "clip"),
                  {"model": "CLIP ViT-B/16 (openai, TINS loader)"})
    del net
    base = load_dino()
    save_features(OUT / "places.dino.pt", ids, encode(paths, dino_transform(), dino_encode_fn(base), guard, 128, 8, "dino"),
                  {"model": "DINOv2 ViT-B/14", "embedding": "x_norm_clstoken"})
    del base
    large = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    large.load_state_dict(torch.load(C.HOME / ".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
                                     map_location="cpu"), strict=True)
    large = large.eval().cuda()
    save_features(OUT / "places.dino_vitl14.pt", ids,
                  encode(paths, dino_transform(), dino_encode_fn(large), guard, 64, 8, "dino_vitl14"),
                  {"model": "DINOv2 ViT-L/14", "embedding": "x_norm_clstoken"})
    meta.to_parquet(OUT / "places_meta.parquet", index=False)
    info = {"n_categories": len(cats), "n_mos_classes": len(mos_cls), "dropped_mos": dropped["mos"],
            "dropped_imagenet": dropped["imagenet"], "n_kept_classes": len(keep),
            "n_classes_dev1": int(sum(1 for i in keep if half[i] == 0)), "n_images": len(meta),
            "n_images_dev1": int((meta.dev == "dev1").sum()), "seconds": round(time.time() - start, 1), "rule": __doc__}
    (OUT / "places_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("rule", "dropped_mos", "dropped_imagenet")}))


if __name__ == "__main__":
    main()
