"""Analyses on the additional benchmarks (post-test; the frozen CLAVIS-M_c1 is applied unchanged).

--parts baselines : MCM / NegLabel on the OpenOOD test streams, alone and multiplied by p (no memory) / p_t (memory)
        fourood   : Four-OOD (ImageNet-1K val vs iNaturalist / SUN / Places / DTD), TINS seeds 123-125
        acrossid  : ImageNet-V2 / -R / -Sketch as ID (TINS across-ID protocol), p from the ImageNet training shots
        l14       : TINS with CLIP ViT-L/14 on the OpenOOD test, alone and multiplied by p_t
The visual view always uses the frozen ImageNet-1K shots (12 support / 4 calibration per class, DINOv2 ViT-B/14).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import score_stream, views_from_arrays  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

OUT = C.WORK / "analysis" / "extra"
FEAT = C.WORK / "extra_feats"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
SEEDS_OO = [123, 124, 125, 126, 127]


def shots():
    blob = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    dino = blob["dino"].numpy().astype(np.float32)
    return blob, dino[:12000].reshape(1000, 12, -1), dino[12000:16000], np.asarray(blob["cand"])[:4000]


def topk(clip_feats, pos, k=5, chunk=8192):
    pos = pos.float()
    out = []
    for lo in range(0, len(clip_feats), chunk):
        out.append((clip_feats[lo:lo + chunk].float() @ pos.T).topk(k, dim=1).indices)
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["baselines", "fourood", "acrossid", "l14"],
                        help="also: acrossid_indomain")
    opts = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    t = import_tins()
    gm = t.get_measures

    def met(s, is_id):
        a, _, f = gm(np.asarray(s[is_id], np.float64), np.asarray(s[~is_id], np.float64))
        return {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}

    blob, support, cal_dino, cal_cand = shots()
    res = {}

    if "baselines" in opts.parts or "l14" in opts.parts:
        dino = blob["dino"].numpy().astype(np.float32)
        queries = dino[12000:]
        is_cal = np.zeros(len(queries), dtype=bool)
        is_cal[:4000] = True
        v = views_from_arrays(support, queries, np.asarray(blob["cand"]), is_cal, m=2)
        ids_all = set()
        for ds in OO:
            ids_all |= set(np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist())
        test_ids = sorted(ids_all)
        row_of = {s: 4000 + i for i, s in enumerate(test_ids)}

        def stream_p(ids, bidx):
            q = np.array([row_of[x] for x in ids])
            p_t, _, _ = score_stream(v, queries[q], v["d"][q], v["p_all"][q], bidx, 0.10, 2, "diff")
            return p_t, v["p"][q]

    if "baselines" in opts.parts:
        bl = np.load(C.WORK / "analysis" / "baselines" / "openood.npz")
        pos_of = {s: i for i, s in enumerate(bl["sample_id"].tolist())}
        per = {}
        for ds in OO:
            for seed in SEEDS_OO:
                run = np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed{seed}.npz")
                ids, is_id = run["sample_id"], ~run["is_ood"].astype(bool)
                p_t, p = stream_p(ids, run["batch_index"])
                k = np.array([pos_of[x] for x in ids])
                r = {}
                for name in ("mcm", "neglabel"):
                    s = bl[name][k].astype(np.float64)
                    r[name] = met(s, is_id)
                    r[f"{name}*p"] = met(s * p, is_id)
                    r[f"{name}*pt"] = met(s * p_t, is_id)
                per[(ds, seed)] = r
        res["baselines_openood"] = {ds: {m: {mm: float(np.mean([per[(ds, s)][m][mm] for s in SEEDS_OO]))
                                             for mm in ("AUROC", "FPR95")} for m in per[(ds, 123)]} for ds in OO}
        print(json.dumps({"baselines": "done", "t": round(time.time() - start)}), flush=True)

    if "l14" in opts.parts:
        per = {}
        for ds in OO:
            run = np.load(C.WORK / "extra_runs" / "l14" / f"l14_{ds}_seed123.npz")
            ref = np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")
            ids, is_id = run["sample_id"], ~run["is_ood"].astype(bool)
            assert np.array_equal(ids, ref["sample_id"]), "L/14 stream order differs from the B/16 seed-123 order"
            bidx = np.arange(len(ids)) // 256
            p_t, p = stream_p(ids, bidx)
            s_l, s_b = run["S_final"].astype(np.float64), ref["S_final"].astype(np.float64)
            per[ds] = {"tins_l14": met(s_l, is_id), "clavism_l14": met(s_l * p_t, is_id), "nomem_l14": met(s_l * p, is_id),
                       "tins_b16": met(s_b, is_id), "clavism_b16": met(s_b * p_t, is_id)}
        res["l14_seed123"] = per
        print(json.dumps({"l14": "done", "t": round(time.time() - start)}), flush=True)

    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"]

    if "fourood" in opts.parts:
        names = ["in_val"] + FOUR
        dn = {n: torch.load(FEAT / f"{n}.dino.pt", map_location="cpu") for n in names}
        cl = {n: torch.load(FEAT / f"{n}.clipb16.pt", map_location="cpu") for n in names}
        ids = sum((dn[n]["ids"] for n in names), [])
        assert ids == sum((cl[n]["ids"] for n in names), [])
        ev = torch.cat([dn[n]["features"] for n in names]).numpy().astype(np.float32)
        cand = topk(torch.cat([cl[n]["features"] for n in names]), pos)
        queries = np.concatenate([cal_dino, ev])
        is_cal = np.zeros(len(queries), dtype=bool)
        is_cal[:4000] = True
        v4 = views_from_arrays(support, queries, np.concatenate([cal_cand, cand]), is_cal, m=2)
        row = {s: 4000 + i for i, s in enumerate(ids)}
        bl = np.load(C.WORK / "analysis" / "baselines" / "fourood.npz")
        bpos = {s: i for i, s in enumerate(bl["sample_id"].tolist())}
        per, conf = {}, {}
        seeds4 = sorted({int(p.stem.split("seed")[1]) for p in (C.WORK / "extra_runs" / "fourood").glob("imagenet_inat_seed*.npz")})
        for ood in FOUR:
            for seed in seeds4:
                run = np.load(C.WORK / "extra_runs" / "fourood" / f"imagenet_{ood}_seed{seed}.npz")
                sid, is_id = run["sample_id"], ~run["is_ood"].astype(bool)
                q = np.array([row[x] for x in sid])
                bidx = run["batch_index"]
                p_t, g, admit = score_stream(v4, queries[q], v4["d"][q], v4["p_all"][q], bidx, 0.10, 2, "diff")
                p = v4["p"][q]
                s = run["S_final"].astype(np.float64)
                cal = run["cal_scores"]
                p_T = np.empty(len(s))
                for b in np.unique(bidx):
                    rr = np.flatnonzero(bidx == b)
                    cs = np.sort(cal[b].astype(np.float64))
                    p_T[rr] = (1.0 + np.searchsorted(cs, s[rr], side="right")) / (len(cs) + 1.0)
                k = np.array([bpos[x] for x in sid])
                r = {"tins": met(s, is_id), "clavism": met(s * p_t, is_id), "nomem": met(s * p, is_id),
                     "fisher_pT_pt": met(p_T * p_t, is_id), "visual_g": met(-g, is_id)}
                for name in ("mcm", "neglabel"):
                    b_ = bl[name][k].astype(np.float64)
                    r[name] = met(b_, is_id)
                    r[f"{name}*pt"] = met(b_ * p_t, is_id)
                per[(ood, seed)] = r
                conf[(ood, seed)] = {f"p_t@{a}": float((p_t[is_id] <= a).mean()) for a in (0.01, 0.05, 0.1)}
        id_rows = np.array([row[s] for s in dn["in_val"]["ids"]])
        res["fourood"] = {ood: {m: {mm: float(np.mean([per[(ood, s)][m][mm] for s in seeds4])) for mm in ("AUROC", "FPR95")}
                                for m in per[(ood, seeds4[0])]} for ood in FOUR}
        res["fourood_sd_over_seeds"] = {ood: {m: float(np.std([per[(ood, s)][m]["AUROC"] for s in seeds4], ddof=1))
                                              for m in ("tins", "clavism")} for ood in FOUR}
        res["fourood_seeds"] = seeds4
        res["fourood_conformal_ID"] = {**{f"p@{a}": float((v4["p"][id_rows] <= a).mean()) for a in (0.01, 0.05, 0.1)},
                                       **{f"p_all@{a}": float((v4["p_all"][id_rows] <= a).mean()) for a in (0.01, 0.05, 0.1)},
                                       **{k_: float(np.mean([conf[(o, s)][k_] for o in FOUR for s in seeds4])) for k_ in conf[(FOUR[0], seeds4[0])]}}
        print(json.dumps({"fourood": "done", "t": round(time.time() - start)}), flush=True)

    if "acrossid" in opts.parts:
        wn1k = sorted(p.name for p in (C.IMAGENET_ROOT / "train").iterdir() if p.is_dir())
        cal_clip = torch.load(C.WORK / "test_runs" / "cal_clip.pt")["features"]
        dn4 = {n: torch.load(FEAT / f"{n}.dino.pt", map_location="cpu") for n in FOUR}
        cl4 = {n: torch.load(FEAT / f"{n}.clipb16.pt", map_location="cpu") for n in FOUR}
        out = {}
        for idn in ("in_v2", "in_r", "in_sketch"):
            setup = torch.load(C.WORK / "extra_runs" / "acrossid" / f"setup_{idn}.pt")
            dn, cl = torch.load(FEAT / f"{idn}.dino.pt", map_location="cpu"), torch.load(FEAT / f"{idn}.clipb16.pt", map_location="cpu")
            if idn == "in_r":
                classes = sorted(set(dn["wnids"]))
                cls1k = np.array([wn1k.index(w) for w in classes])
            else:
                cls1k = np.arange(1000)
            sup = support[cls1k]
            cal_idx = np.concatenate([np.arange(c * 4, c * 4 + 4) for c in cls1k])
            test = setup["test"]
            ev_ids = [dn["ids"][i] for i in test] + sum((dn4[n]["ids"] for n in FOUR), [])
            ev = np.concatenate([dn["features"].numpy()[test]] + [dn4[n]["features"].numpy() for n in FOUR]).astype(np.float32)
            ev_clip = torch.cat([cl["features"][test]] + [cl4[n]["features"] for n in FOUR])
            cand = topk(ev_clip, setup["pos"])
            cand_cal = topk(cal_clip[cal_idx], setup["pos"])
            queries = np.concatenate([cal_dino[cal_idx], ev])
            is_cal = np.zeros(len(queries), dtype=bool)
            is_cal[:len(cal_idx)] = True
            va = views_from_arrays(sup, queries, np.concatenate([cand_cal, cand]), is_cal, m=2)
            row = {s: len(cal_idx) + i for i, s in enumerate(ev_ids)}
            id_q = np.array([row[s] for s in [dn["ids"][i] for i in test]])
            r = {"conformal_ID": {**{f"p@{a}": float((va["p"][id_q] <= a).mean()) for a in (0.01, 0.05, 0.1)},
                                  **{f"p_all@{a}": float((va["p_all"][id_q] <= a).mean()) for a in (0.01, 0.05, 0.1)}},
                 "n_classes": int(len(cls1k)), "n_test_ID": int(len(test))}
            per = {}
            pts = []
            for ood in FOUR:
                run = np.load(C.WORK / "extra_runs" / "acrossid" / f"{idn}_{ood}_seed123.npz")
                sid, is_id = run["sample_id"], ~run["is_ood"].astype(bool)
                q = np.array([row[x] for x in sid])
                bidx = np.arange(len(sid)) // 256
                p_t, g, admit = score_stream(va, queries[q], va["d"][q], va["p_all"][q], bidx, 0.10, 2, "diff")
                s = run["S_final"].astype(np.float64)
                per[ood] = {"tins": met(s, is_id), "clavism": met(s * p_t, is_id), "nomem": met(s * va["p"][q], is_id)}
                pts.append({f"p_t@{a}": float((p_t[is_id] <= a).mean()) for a in (0.01, 0.05, 0.1)})
            r["per_ood"] = per
            r["mean"] = {m: {mm: float(np.mean([per[o][m][mm] for o in FOUR])) for mm in ("AUROC", "FPR95")}
                         for m in ("tins", "clavism", "nomem")}
            r["conformal_ID"].update({k_: float(np.mean([x[k_] for x in pts])) for k_ in pts[0]})
            out[idn] = r
            print(json.dumps({"acrossid": idn, "mean": r["mean"], "conf": r["conformal_ID"]}), flush=True)
        res["acrossid"] = out

    if "acrossid_indomain" in opts.parts:
        # same protocol, but the visual view's support (3) and calibration (1) come from TINS's 4 in-domain proxies
        dn4 = {n: torch.load(FEAT / f"{n}.dino.pt", map_location="cpu") for n in FOUR}
        cl4 = {n: torch.load(FEAT / f"{n}.clipb16.pt", map_location="cpu") for n in FOUR}
        out = {}
        for idn in ("in_v2", "in_r", "in_sketch"):
            setup = torch.load(C.WORK / "extra_runs" / "acrossid" / f"setup_{idn}.pt")
            dn, cl = torch.load(FEAT / f"{idn}.dino.pt", map_location="cpu"), torch.load(FEAT / f"{idn}.clipb16.pt", map_location="cpu")
            targets, n_cls = np.asarray(setup["targets"]), len(setup["labels"])
            by = [[i for i in setup["proxy"] if targets[i] == c] for c in range(n_cls)]
            assert all(len(b) == 4 for b in by)
            feats = dn["features"].numpy().astype(np.float32)
            sup = np.stack([feats[b[:3]] for b in by])
            cal_rows = [b[3] for b in by]
            test = setup["test"]
            ev_ids = [dn["ids"][i] for i in test] + sum((dn4[n]["ids"] for n in FOUR), [])
            ev = np.concatenate([feats[test]] + [dn4[n]["features"].numpy() for n in FOUR]).astype(np.float32)
            ev_clip = torch.cat([cl["features"][test]] + [cl4[n]["features"] for n in FOUR])
            cand = topk(ev_clip, setup["pos"])
            cand_cal = topk(cl["features"][cal_rows], setup["pos"])
            queries = np.concatenate([feats[cal_rows], ev])
            is_cal = np.zeros(len(queries), dtype=bool)
            is_cal[:len(cal_rows)] = True
            va = views_from_arrays(sup, queries, np.concatenate([cand_cal, cand]), is_cal, m=2, n0=3)
            row = {s_: len(cal_rows) + i for i, s_ in enumerate(ev_ids)}
            id_q = np.array([row[s_] for s_ in [dn["ids"][i] for i in test]])
            r = {"conformal_ID": {**{f"p@{a}": float((va["p"][id_q] <= a).mean()) for a in (0.01, 0.05, 0.1)},
                                  **{f"p_all@{a}": float((va["p_all"][id_q] <= a).mean()) for a in (0.01, 0.05, 0.1)}}}
            per, pts = {}, []
            for ood in FOUR:
                run = np.load(C.WORK / "extra_runs" / "acrossid" / f"{idn}_{ood}_seed123.npz")
                sid, is_id = run["sample_id"], ~run["is_ood"].astype(bool)
                q = np.array([row[x] for x in sid])
                bidx = np.arange(len(sid)) // 256
                p_t, g, admit = score_stream(va, queries[q], va["d"][q], va["p_all"][q], bidx, 0.10, 2, "diff")
                s = run["S_final"].astype(np.float64)
                per[ood] = {"tins": met(s, is_id), "clavism": met(s * p_t, is_id), "nomem": met(s * va["p"][q], is_id)}
                pts.append({f"p_t@{a}": float((p_t[is_id] <= a).mean()) for a in (0.01, 0.05, 0.1)})
            r["per_ood"] = per
            r["mean"] = {m: {mm: float(np.mean([per[o][m][mm] for o in FOUR])) for mm in ("AUROC", "FPR95")}
                         for m in ("tins", "clavism", "nomem")}
            r["conformal_ID"].update({k_: float(np.mean([x[k_] for x in pts])) for k_ in pts[0]})
            out[idn] = r
            print(json.dumps({"acrossid_indomain": idn, "mean": r["mean"], "conf": r["conformal_ID"]}), flush=True)
        res["acrossid_indomain"] = out

    prev = json.loads((OUT / "analysis_extra.json").read_text()) if (OUT / "analysis_extra.json").exists() else {}
    prev.update(res)
    prev["seconds_last"] = round(time.time() - start, 1)
    (OUT / "analysis_extra.json").write_text(json.dumps(prev, indent=1, default=str) + "\n")
    print(json.dumps({"done": list(res)}))


if __name__ == "__main__":
    main()
