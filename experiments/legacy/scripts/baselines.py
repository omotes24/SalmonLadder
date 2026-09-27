"""Static zero-shot baselines on the same CLIP ViT-B/16 features as TINS: MCM and NegLabel.

MCM      : max_c softmax(cos(v, t_c) / T), T = 1, TINS's positive prompts
NegLabel : the reference implementation shipped with TINS (eval_neglabel_ood.py): prompts "the nice {}" /
           "This is a {} photo", negatives ranked by the 95th percentile similarity to the ID labels, 15 % of nouns and
           adjectives kept, 100 groups, T = 1, logit scale 100, tail dropped then a fixed permutation (seed 0);
           text features from the OpenAI CLIP text encoder (as TINS) instead of the HuggingFace port.
Also saves TINS's positive text features (zero-shot candidates for the extra benchmarks).
Output: <WORK>/analysis/baselines/{openood,fourood}.npz, pos_tins.pt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.tins_dev import get_logger, import_tins, load_clip  # noqa: E402
from scripts.run_tins_test import codex_args, load_stream_cache  # noqa: E402

OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["in_val", "inat", "sun", "places", "dtd"]


@torch.no_grad()
def mcm(feats, pos, device, T=1.0, chunk=8192):
    out = []
    for lo in range(0, len(feats), chunk):
        sims = feats[lo:lo + chunk].to(device) @ pos.T
        out.append((sims / T).softmax(dim=-1).max(dim=-1).values.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def neglabel(feats, pos, neg, device, ngroup=100, t=1.0, scale=100.0, chunk=4096):
    eff = neg.shape[0] - neg.shape[0] % ngroup
    neg = neg[:eff]
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    perm = torch.randperm(eff, device=device)
    out = []
    for lo in range(0, len(feats), chunk):
        b = feats[lo:lo + chunk].to(device)
        pl, nl = scale * (b @ pos.T), scale * (b @ neg.T)
        grouped = nl[:, perm].reshape(b.shape[0], ngroup, -1)
        sc = []
        for g in range(ngroup):
            logits = torch.cat([pl, grouped[:, g, :]], dim=-1) / t
            sc.append(logits.softmax(dim=-1)[:, :pl.shape[1]].sum(dim=-1, keepdim=True))
        out.append(torch.cat(sc, dim=-1).mean(dim=-1).cpu())
    return torch.cat(out).numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["openood", "fourood"])
    opts = parser.parse_args()
    out = C.WORK / "analysis" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    t = import_tins()
    args = codex_args(t, C.WORK / "test_runs" / "tins_cache", "vins_baselines", 123)
    log = get_logger(out / "run.log")
    net, _ = load_clip(t, args)
    device = next(net.parameters()).device
    labels = [str(x) for x in t.get_test_labels(args, None)]
    pos = t.encode_texts(net, [args.pos_prompt.format(l) for l in labels], batch_size=1000, device=device,
                         desc="pos").to(device).float()
    pos = pos / pos.norm(dim=-1, keepdim=True)
    torch.save({"labels": labels, "pos": pos.cpu(), "prompt": args.pos_prompt}, out / "pos_tins.pt")
    # NegLabel label bank (reference algorithm, OpenAI CLIP text encoder)
    wordnet = t.resolve_wordnet_dir(args)
    dedup, nouns, adjs = set(), [], []
    for path in sorted(Path(wordnet).glob("*.txt")):
        kind = path.name.split(".")[0]
        if kind not in {"noun", "adj"}:
            continue
        for line in path.read_text().splitlines():
            w = line.strip()
            if not w or w in dedup:
                continue
            dedup.add(w)
            (nouns if kind == "noun" else adjs).append(("the nice {}" if kind == "noun" else "This is a {} photo").format(w))
    pos_nl = t.encode_texts(net, ["the nice {}".format(l) for l in labels], batch_size=1000, device=device,
                            desc="pos-nl").to(device).float()
    pos_nl = pos_nl / pos_nl.norm(dim=-1, keepdim=True)
    nf = t.encode_texts(net, nouns, batch_size=1000, device=device, desc="nouns").float()
    af = t.encode_texts(net, adjs, batch_size=1000, device=device, desc="adjs").float()
    nf, af = nf / nf.norm(dim=-1, keepdim=True), af / af.norm(dim=-1, keepdim=True)
    allf = torch.cat([nf, af]).to(device)
    rank = torch.cat([torch.quantile((allf[i:i + 1000] @ pos_nl.T).float(), q=0.95, dim=-1)
                      for i in range(0, len(allf), 1000)]).cpu()
    n_keep, a_keep = max(1, int(len(nouns) * 0.15)), max(1, int(len(adjs) * 0.15))
    ni = torch.argsort(rank[:len(nouns)])[:n_keep]
    ai = torch.argsort(rank[len(nouns):])[:a_keep]
    neg = torch.cat([nf[ni], af[ai]]).to(device)
    log.debug(f"NegLabel negatives: {len(ni)} nouns + {len(ai)} adjectives")

    ids = []
    if "openood" in opts.parts:
        feat_of = {}
        for ds in OO:
            sid, _, f, _ = load_stream_cache(ds)
            for i, s_ in enumerate(sid):
                if s_ not in feat_of:
                    feat_of[s_] = f[i]
        ids = sorted(feat_of)
        feats = torch.stack([feat_of[s_] for s_ in ids])
        np.savez_compressed(out / "openood.npz", sample_id=np.array(ids), mcm=mcm(feats, pos, device),
                            neglabel=neglabel(feats, pos_nl, neg, device))
    ids4, sc_m, sc_n = [], [], []
    for name in (FOUR if "fourood" in opts.parts else []):
        blob = torch.load(C.WORK / "extra_feats" / f"{name}.clipb16.pt", map_location="cpu")
        ids4 += blob["ids"]
        sc_m.append(mcm(blob["features"], pos, device))
        sc_n.append(neglabel(blob["features"], pos_nl, neg, device))
    if ids4:
        np.savez_compressed(out / "fourood.npz", sample_id=np.array(ids4), mcm=np.concatenate(sc_m),
                            neglabel=np.concatenate(sc_n))
    print(json.dumps({"openood": len(ids), "fourood": len(ids4), "neg": int(neg.shape[0])}))


if __name__ == "__main__":
    main()
