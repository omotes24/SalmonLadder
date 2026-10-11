"""Phase 15 (b): baselines that use pre-trained features with many labels, on the OpenOOD v1.5 ImageNet-200 and CIFAR-100
benchmarks (the same test sets as in200_14.py / cifar14.py):
  knn_all_k50, knn_all_k10 : cosine similarity to the k-th nearest TRAINING image over all training images (Sun et al. 2022;
                             OpenOOD's KNN uses K = 50), features = the paper's two views concatenated after L2 normalisation
                             (the cosine in the concatenation is the mean of the per-view cosines); also per single view (k = 50).
  ncm_all                  : maximum cosine to the class means of all training images (same features).
  mcm_clip                 : zero-shot MCM (Ming et al. 2022): CLIP ViT-B/16, prompt 'a photo of a {}.', softmax of the cosine
                             similarities at temperature 1, maximum over the ID classes (as measured in Phase 11 on ImageNet-1K).
  neglabel_clip            : zero-shot NegLabel (Jiang et al. 2024): the official OpenOOD-VLM post-processor (NegLabel = AdaNeg's
                             vanilla score) with the official text side built for the benchmark's class names (dataset
                             'imagenet200' / 'cifar100' of the official code; 10,000 negative labels, 'nice' prompt, text_center).
Image features: ImageNet-200 from the Phase 3 / 10 ImageNet-1K spaces (the OOD lists are the ImageNet-1K ones; the ID test images are
the OpenOOD ImageNet-200 validation list), CIFAR-100 from the Phase 14 table (DINO views) and extra15 (CLIP).
Usage: --bench imagenet200|cifar100 [--worker w --nworkers n] | --text imagenet200|cifar100 (build the text side only)
Output: P15/results/base/<bench>/<ds>.csv (method, AUROC, FPR95), <ds>.meta.json."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p15common as P
from an11 import measures
from draws12 import datasets
from in200_14 import layout
from extract14 import CIFAR, OOD as CIFAR_OOD

sys.path.insert(0, str(P.P4 / "code"))
import vlm_tta as VT  # noqa: E402

TABLES, FEATS = P.P15 / "tables", P.P15 / "features"
OUT = P.RESULTS15 / "base"
TEXT = P.P15 / "text"
VIEWS = ["L14", "D3L"]
KS = (50, 10)
MCM_TEMPLATE = "a photo of a {}."


def build_text(cfp, dataset):
    """Official FixedCLIP_NegOODPrompt text side for the benchmark's class names (cached), plus the MCM prompt features."""
    TEXT.mkdir(parents=True, exist_ok=True)
    f = TEXT / f"text_{dataset}.pt"
    if f.exists():
        d = torch.load(f, map_location="cuda")
        s = VT.Cached()
        for k, v in d.items():
            setattr(s, k, v)
        s.__class__ = VT.StreamNet
        return s, d["mcm_text"], d["names"]
    t0 = time.time()
    cwd = os.getcwd()
    os.chdir(VT.VT)
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        net = cfp.FixedCLIP_NegOODPrompt(VT.A(backbone=VT.A(name="ViT-B/16", dataset=dataset, text_prompt="nice", text_center=True, ood_number=10000)))
        names = list(cfp.get_class_names(dataset))
        with torch.no_grad():
            toks = cfp.clip.tokenize([MCM_TEMPLATE.format(n.replace("_", " ")) for n in names]).cuda()
            mcm = net.model.encode_text(toks).float()
            mcm = mcm / mcm.norm(dim=-1, keepdim=True)
    finally:
        os.chdir(cwd)
    s = VT.StreamNet(net)
    del net
    torch.cuda.empty_cache()
    d = {k: getattr(s, k) for k in ("text_features", "text_features_unselected", "text_features_all", "noise_image_features", "logit_scale", "n_cls", "n_output")}
    d.update(dataset=dataset, names=names, mcm_text=mcm, setup_s=time.time() - t0, utc=P.utc())
    tmp = str(f) + f".{os.getpid()}"
    torch.save(d, tmp)
    os.replace(tmp, f)
    print(json.dumps({"text_ready": dataset, "n_cls": int(s.n_cls), "n_neg": int(s.text_features.shape[1] - s.n_cls), "setup_s": round(d["setup_s"], 1)}), flush=True)
    return s, mcm, names


def mcm_scores(clip_feats, mcm_text):
    X = torch.as_tensor(np.asarray(clip_feats, np.float32), device="cuda")
    X = X / X.norm(dim=-1, keepdim=True)
    out = []
    for lo in range(0, len(X), 8192):
        sims = (X[lo:lo + 8192] @ mcm_text.T).double()
        out.append(torch.softmax(sims, dim=1).max(1).values.cpu().numpy())
    return np.concatenate(out)


def neglabel_scores(snet, clip_feats, classes):
    kind, args = VT.METHODS["NegLabel"]
    sc, _, _ = VT.run_stream(snet, np.asarray(clip_feats, np.float32), kind, args, classes, 0)
    assert np.isfinite(sc).all()
    return sc


class Bank:
    """All training features of one benchmark on the GPU (float16), per view and concatenated; class means."""

    def __init__(self, feats, labels, dev="cuda"):
        self.dev = dev
        self.F = {v: torch.as_tensor(np.asarray(f, np.float32), device=dev).half() for v, f in feats.items()}
        cat = torch.cat([self.F[v].float() for v in VIEWS], dim=1) / np.sqrt(len(VIEWS))
        self.F["cat"] = cat.half()
        y = torch.as_tensor(np.asarray(labels), device=dev)
        self.means = {}
        for v, F in self.F.items():
            M = torch.zeros((int(y.max()) + 1, F.shape[1]), dtype=torch.float32, device=dev).index_add_(0, y, F.float())
            self.means[v] = M / M.norm(dim=1, keepdim=True)

    def knn(self, X, view, ks):
        F = self.F[view]
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.dev).half()
        kmax = max(ks)
        tops = []
        for lo in range(0, len(X), 2048):
            s = X[lo:lo + 2048] @ F.T
            tops.append(s.float().topk(kmax, dim=1).values)
        T = torch.cat(tops)
        return {k: T[:, k - 1].double().cpu().numpy() for k in ks}

    def ncm(self, X, view):
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.dev)
        return torch.cat([(X[lo:lo + 8192] @ self.means[view].T).max(1).values for lo in range(0, len(X), 8192)]).double().cpu().numpy()


def test_feats(views_feats, flag):
    cat = np.concatenate([P.l2n(views_feats[v]) for v in VIEWS], axis=1) / np.sqrt(len(VIEWS))
    return {**{v: P.l2n(views_feats[v]) for v in VIEWS}, "cat": cat}


def score_sets(bank, sets, clip_of, snet, mcm_text, classes, bench, out_dir):
    """sets: {ds: (group, flag, {view: test features}, clip features)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, (group, flag, vf, cf) in sets.items():
        if (out_dir / f"{ds}.csv").exists():
            continue
        t0 = time.time()
        X = test_feats(vf, flag)
        rows = []

        def add(method, x, extra=None):
            au, fp = measures(np.asarray(x, np.float64), flag)
            rows.append({"benchmark": bench, "ds": ds, "group": group, "method": method, "AUROC": au, "FPR95": fp, "n": int(len(flag)), "n_ood": int(flag.sum()), **(extra or {})})
        kn = bank.knn(X["cat"], "cat", KS)
        for k in KS:
            add(f"knn_all_k{k}", kn[k], {"features": "L14+D3L", "k": k})
        for v in VIEWS:
            add(f"knn_all_k50_{v}", bank.knn(X[v], v, (50,))[50], {"features": v, "k": 50})
        add("ncm_all", bank.ncm(X["cat"], "cat"), {"features": "L14+D3L"})
        for v in VIEWS:
            add(f"ncm_all_{v}", bank.ncm(X[v], v), {"features": v})
        if cf is not None:
            add("mcm_clip", mcm_scores(cf, mcm_text), {"features": "CLIP ViT-B/16"})
            add("neglabel_clip", neglabel_scores(snet, cf, classes), {"features": "CLIP ViT-B/16"})
        pd.DataFrame(rows).to_csv(out_dir / f"{ds}.csv", index=False)
        (out_dir / f"{ds}.meta.json").write_text(json.dumps({"benchmark": bench, "ds": ds, "group": group, "n": int(len(flag)), "n_ood": int(flag.sum()),
                                                            "n_train": int(len(bank.F["cat"])), "seconds": round(time.time() - t0, 1), "utc": P.utc()}, indent=1) + "\n")
        print(json.dumps({"bench": bench, "ds": ds, "FPR95": {r["method"]: round(r["FPR95"], 2) for r in rows}, "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


def run_imagenet200(worker, nworkers, snet, mcm_text, classes):
    t = pd.read_parquet(TABLES / "in200_train.parquet")
    feats = {v: np.load(FEATS / "in200_train" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    bank = Bank({v: feats[v] for v in VIEWS}, t.label.values)
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")
    id_ids, _, _ = layout(D, paths)
    all_sets = datasets("openood")
    groups = {"ssb_hard": "near", "ninco": "near", "inaturalist": "far", "textures": "far", "openimageo": "far"}
    sets = {}
    for ds in sorted(all_sets)[worker::nworkers]:
        ids_all, flag_all, _ = all_sets[ds]
        ood_ids = ids_all[flag_all]
        test_ids = np.concatenate([id_ids, ood_ids])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids]) - 4000
        sets[ds] = (groups[ds], flag, {v: np.asarray(D["ev"][v][rows], np.float32) for v in VIEWS}, np.asarray(D["ev"]["CLIP"][rows], np.float32))
    score_sets(bank, sets, None, snet, mcm_text, classes, "imagenet200", OUT / "imagenet200")


def run_cifar100(worker, nworkers, snet, mcm_text, classes):
    ttr = pd.read_parquet(TABLES / "cifar_train.parquet")
    feats = {v: np.load(FEATS / "cifar_train" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    bank = Bank({v: feats[v] for v in VIEWS}, ttr.label.values)
    t14 = pd.read_parquet(CIFAR / "table.parquet")
    F14 = {v: np.load(CIFAR / "features" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    tcl = pd.read_parquet(TABLES / "cifar_clip.parquet")
    Fcl = np.load(FEATS / "cifar_clip" / "CLIP.npy", mmap_mode="r")
    clip_pos = {int(r): i for i, r in enumerate(tcl.p14_row.values)}
    id_rows = t14.index[t14.split == "test"].values
    sets = {}
    for ds in sorted(CIFAR_OOD)[worker::nworkers]:
        ood_rows = t14.index[t14.dataset == ds].values
        rows = np.concatenate([id_rows, ood_rows])
        flag = np.r_[np.zeros(len(id_rows), bool), np.ones(len(ood_rows), bool)]
        crow = np.array([clip_pos[int(r)] for r in rows])
        sets[ds] = (CIFAR_OOD[ds], flag, {v: np.asarray(F14[v][rows], np.float32) for v in VIEWS}, np.asarray(Fcl[crow], np.float32))
    score_sets(bank, sets, None, snet, mcm_text, classes, "cifar100", OUT / "cifar100")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["imagenet200", "cifar100"], default=None)
    ap.add_argument("--text", choices=["imagenet200", "cifar100"], default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfp, Ada, Tanl = VT.import_official()
    classes = {"ada": Ada, "tanl": Tanl}
    ds = a.text or a.bench
    snet, mcm_text, names = build_text(cfp, ds)
    if a.bench == "imagenet200":
        run_imagenet200(a.worker, a.nworkers, snet, mcm_text, classes)
    elif a.bench == "cifar100":
        run_cifar100(a.worker, a.nworkers, snet, mcm_text, classes)
    print("BASE15_DONE", flush=True)
