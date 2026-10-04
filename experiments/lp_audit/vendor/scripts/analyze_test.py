"""Post-test analyses on the OpenOOD v1.5 ImageNet-1K test. The methods are frozen; nothing here selects a method.

Inputs : test_runs/default/<ds>_seed<s>.npz (TINS, 5 stream orders), test_runs/permfirst, combined streams,
         test_eval/results/features.pt (DINOv2 support / calibration / test, CLIP zero-shot candidates), Codex manifest
Outputs: analysis/test/analysis.json and report.md
Scores : tins, clavism (S*p_t), nomem (S*p, CLAVIS-M without memory), visual scores, fusion-scale variants
         (S^a*p_t, p_T*p_t = Fisher, min(p_T, p_t)), plain DINOv2 kNN (S*p_knn), pooled standardisation (static / memory)
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.candidates import class_r_all  # noqa: E402
from vins.clavism import score_stream, views_from_arrays  # noqa: E402
from vins.memory import pvalues_sorted  # noqa: E402
from vins.stats import auroc, delong_paired, draw_members, precompute_members  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
GROUP = {"ssb_hard": "near", "ninco": "near", "inaturalist": "far", "textures": "far", "openimageo": "far"}
SEEDS = [123, 124, 125, 126, 127]
PUB = {"ssb_hard": (81.38, 61.68), "ninco": (83.91, 54.09), "inaturalist": (99.93, 0.21), "textures": (97.83, 10.09),
       "openimageo": (92.99, 28.68)}
R = C.WORK / "test_runs"
OUT = C.WORK / "analysis" / "test"
G = {}   # globals shared with bootstrap workers (fork)


def p_from_cal(cal_rows, bidx, s):
    """TINS conformal p-value: rank of S among the calibration shots scored under the same negatives."""
    out = np.empty(len(s))
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        cal = np.sort(cal_rows[b].astype(np.float64))
        out[rows] = (1.0 + np.searchsorted(cal, s[rows], side="right")) / (len(cal) + 1.0)
    return out


def kth_nn_all(queries, support_flat, k=2, chunk=4096):
    out = np.empty(len(queries))
    for lo in range(0, len(queries), chunk):
        sims = queries[lo:lo + chunk] @ support_flat.T
        top = -np.partition(-sims, k - 1, axis=1)[:, :k]
        out[lo:lo + chunk] = 1.0 - np.sort(top, axis=1)[:, 0]
    return out


def boot_worker(seed):
    rng = np.random.default_rng(seed)
    idc = rng.integers(0, G["n_id_cls"], G["n_id_cls"])
    res = {}
    for ds in OO:
        st = G["streams"][ds]
        id_pos = np.concatenate([st["id_members"][c] for c in idc])
        ood_pos = draw_members(st["ood_pre"], rng)
        for m in ("tins", "clavism", "nomem"):
            s = st["scores"][m]
            res[(ds, m)] = auroc(s[id_pos], s[ood_pos])
    return res


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    t = import_tins()
    gm = t.get_measures

    def met(s, is_id):
        a, _, f = gm(s[is_id].astype(np.float64), s[~is_id].astype(np.float64))
        return {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}

    blob = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    dino = blob["dino"].numpy().astype(np.float32)
    n_sup, n_cal = 1000 * C.N_SUPPORT, 1000 * C.N_CALIB
    support = dino[:n_sup].reshape(1000, C.N_SUPPORT, -1)
    queries = dino[n_sup:]
    cand = np.asarray(blob["cand"])
    is_cal = np.zeros(len(queries), dtype=bool)
    is_cal[:n_cal] = True
    v = views_from_arrays(support, queries, cand, is_cal, m=2)
    st_ = v["stats"]
    # ablation views: pooled standardisation (no class-wise scale) and plain kNN over all support images
    r_all = class_r_all(queries.astype(np.float64), support.astype(np.float64), 2)
    r_k = np.take_along_axis(r_all, cand, axis=1)
    d_pooled = (r_k.min(axis=1) - st_["med_all"]) / st_["mad_all"]
    z_k = (r_k - st_["med_t"][cand]) / st_["mad_t"][cand]
    cstar = cand[np.arange(len(cand)), z_k.argmin(axis=1)]
    d_all_raw = r_all.min(axis=1)
    del r_all
    d_knn = kth_nn_all(queries, support.reshape(-1, support.shape[-1]), k=2)
    p_pooled = pvalues_sorted(np.sort(d_pooled[is_cal]), d_pooled)
    p_knn = pvalues_sorted(np.sort(d_knn[is_cal]), d_knn)
    v_pool = dict(v, d_cal=d_pooled[is_cal])

    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    ids_all = set()
    for ds in OO:
        ids_all |= set(np.load(R / "default" / f"{ds}_seed123.npz")["sample_id"].tolist())
    test_ids = sorted(ids_all)
    assert len(test_ids) == len(queries) - n_cal
    row_of = {s: n_cal + i for i, s in enumerate(test_ids)}

    def cls_of(sid):
        rel = manifest[sid]["relative_path"]
        if sid.startswith("imagenet"):
            return f"id{manifest[sid]['label']}"
        parts = rel.split("/")
        return parts[1] if len(parts) > 2 and not parts[1] == "images" else sid

    results, diag, conf = {}, {}, {}
    near_keep = {}
    for ds in OO:
        for seed in SEEDS:
            run = np.load(R / "default" / f"{ds}_seed{seed}.npz")
            ids, is_ood = run["sample_id"], run["is_ood"].astype(bool)
            is_id = ~is_ood
            s = run["S_final"].astype(np.float64)
            bidx = run["batch_index"]
            q = np.array([row_of[x] for x in ids])
            p_t_, g, admit = score_stream(v, queries[q], v["d"][q], v["p_all"][q], bidx, 0.10, 2, "diff")
            p_tp, _, _ = score_stream(v_pool, queries[q], d_pooled[q], v["p_all"][q], bidx, 0.10, 2, "diff")
            p_T = p_from_cal(run["cal_scores"], bidx, s)
            p = v["p"][q]
            sc = {"tins": s, "clavism": s * p_t_, "nomem": s * p, "visual_g": -g, "visual_d": -v["d"][q],
                  "fisher_pT_pt": p_T * p_t_, "min_pT_pt": np.minimum(p_T, p_t_), "fisher_pT_p": p_T * p,
                  "tins_pT": p_T, "knn_static": s * p_knn[q], "visual_knn": -d_knn[q],
                  "pooled_static": s * p_pooled[q], "pooled_mem": s * p_tp, "visual_pooled": -d_pooled[q],
                  "visual_dall": -d_all_raw[q]}
            for a in (0.25, 0.5, 2.0, 4.0):
                sc[f"S^{a:g}*pt"] = s ** a * p_t_
            results[(ds, seed)] = {k: met(val, is_id) for k, val in sc.items()}
            # diagnostics
            seeded = run["seeded"].astype(bool)
            inv = run["inv_pass"]
            k = np.minimum(run["bank_size"], 2000)
            dropped = (2000 + k) % 5
            diag[(ds, seed)] = {
                "P_seed_ID": float(seeded[is_id].mean()), "P_seed_OOD": float(seeded[is_ood].mean()),
                "pass_ID": float((inv[is_id & seeded] == 1).mean()) if (is_id & seeded).any() else None,
                "pass_OOD": float((inv[is_ood & seeded] == 1).mean()) if (is_ood & seeded).any() else None,
                "accepted_from_ID": int(((inv == 1) & is_id).sum()), "accepted_from_OOD": int(((inv == 1) & is_ood).sum()),
                "learned_dropped_frac": float(dropped.sum() / max(1, k.sum())),
                "batches_all_learned_dropped": int(((k > 0) & (k < 5)).sum()),
                "first_batch_bank_ge5": int(np.argmax(k >= 5)) if (k >= 5).any() else None,
                "n_batches": int(len(k)), "final_bank": int(run["bank_size"][-1]),
                "admit_ID": float(admit[is_id].mean()), "admit_OOD": float(admit[is_ood].mean())}
            conf[(ds, seed)] = {f"{name}@{a}": float((pv[is_id] <= a).mean()) for name, pv in
                                (("p_t", p_t_), ("p_T", p_T)) for a in (0.01, 0.05, 0.1)}
            if seed == 123:
                near_keep[ds] = {"ids": ids, "is_ood": is_ood, "scores": {m: sc[m] for m in ("tins", "clavism", "nomem")},
                                 "q": q, "bidx": bidx, "admit": admit}
            print(json.dumps({"ds": ds, "seed": seed, "tins": results[(ds, seed)]["tins"]["AUROC"],
                              "clavism": results[(ds, seed)]["clavism"]["AUROC"], "t": round(time.time() - start)}),
                  flush=True)

    # static p on the 45,000 ID test images
    id_rows = np.array([row_of[s] for s in test_ids if s.startswith("imagenet")])
    conf_static = {f"{name}@{a}": float((pv[id_rows] <= a).mean()) for name, pv in
                   (("p", v["p"]), ("p_all", v["p_all"]), ("p_knn", p_knn), ("p_pooled", p_pooled)) for a in (0.01, 0.05, 0.1)}

    # permute-first variant and combined streams (seed 123 / seeds 123-125)
    perm = {}
    for ds in OO:
        run = np.load(R / "permfirst" / f"{ds}_seed123.npz")
        is_id = ~run["is_ood"].astype(bool)
        perm[ds] = met(run["S_final"].astype(np.float64), is_id)
    combined = {}
    for name, members in (("near_all", ["ssb_hard", "ninco"]), ("far_all", ["inaturalist", "textures", "openimageo"])):
        for seed in (123, 124, 125):
            run = np.load(R / "default" / f"{name}_seed{seed}.npz")
            ids, s = run["sample_id"], run["S_final"].astype(np.float64)
            is_id = np.array([x.startswith("imagenet") for x in ids])
            for ds in members:
                pre = {"ssb_hard": "ssb_hard", "ninco": "ninco", "inaturalist": "inaturalist", "textures": "textures",
                       "openimageo": "openimageo"}[ds]
                sel = np.array([x.rsplit("_", 1)[0] == pre for x in ids])
                a, _, f = gm(s[is_id], s[sel])
                combined.setdefault(ds, []).append({"AUROC": 100 * float(a), "FPR95": 100 * float(f)})

    # DeLong (paired) per dataset and seed
    delong = {}
    for ds in OO:
        k_ = near_keep[ds]
        io = k_["is_ood"]
        sc = k_["scores"]
        for a_, b_ in (("clavism", "tins"), ("clavism", "nomem"), ("nomem", "tins")):
            aa, bb, z, pv = delong_paired(sc[a_][~io], sc[a_][io], sc[b_][~io], sc[b_][io])
            delong[f"{ds}|{a_}-{b_}"] = {"auc_a": 100 * aa, "auc_b": 100 * bb, "z": z, "p": pv}

    # class-level bootstrap (seed 123): ID classes shared across streams, OOD classes per dataset
    G["n_id_cls"] = 1000
    G["streams"] = {}
    for ds in OO:
        k_ = near_keep[ds]
        io = k_["is_ood"]
        labels = np.array([cls_of(x) for x in k_["ids"]])
        id_pos = np.flatnonzero(~io)
        id_lab = np.array([int(labels[i][2:]) for i in id_pos])
        members = [id_pos[id_lab == c] for c in range(1000)]
        ood_pos = np.flatnonzero(io)
        order, starts, counts = precompute_members(labels[ood_pos])
        G["streams"][ds] = {"id_members": members, "ood_pre": (ood_pos[order], starts, counts),
                            "scores": k_["scores"], "n_ood_cls": int(len(counts))}
    B = 1000
    with mp.get_context("fork").Pool(16) as pool:
        reps = pool.map(boot_worker, range(B))
    boot = {}
    for ds in OO:
        for m in ("tins", "clavism", "nomem"):
            arr = np.array([r[(ds, m)] for r in reps]) * 100
            boot[f"{ds}|{m}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        for a_, b_ in (("clavism", "tins"), ("clavism", "nomem")):
            arr = np.array([r[(ds, a_)] - r[(ds, b_)] for r in reps]) * 100
            boot[f"{ds}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
    for grp, names in (("near", ["ssb_hard", "ninco"]), ("far", ["inaturalist", "textures", "openimageo"])):
        for m in ("tins", "clavism", "nomem"):
            arr = np.array([np.mean([r[(n, m)] for n in names]) for r in reps]) * 100
            boot[f"{grp}|{m}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        for a_, b_ in (("clavism", "tins"), ("clavism", "nomem")):
            arr = np.array([np.mean([r[(n, a_)] - r[(n, b_)] for n in names]) for r in reps]) * 100
            boot[f"{grp}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
    n_cls = {ds: G["streams"][ds]["n_ood_cls"] for ds in OO}

    # stream-position quartiles (seed 123) and memory recurrence
    position, recurrence = {}, {}
    for ds in ("ssb_hard", "ninco", "textures"):
        k_ = near_keep[ds]
        io, n = k_["is_ood"], len(k_["is_ood"])
        quart = np.minimum((np.arange(n) * 4) // n, 3)
        position[ds] = {}
        for qq in range(4):
            sel = quart == qq
            position[ds][qq] = {m: 100 * auroc(k_["scores"][m][sel & ~io], k_["scores"][m][sel & io])
                                for m in ("tins", "clavism", "nomem")}
        labels = np.array([cls_of(x) for x in k_["ids"]])
        seen, prior = set(), np.zeros(n, dtype=bool)
        for b in np.unique(k_["bidx"]):
            rows = np.flatnonzero(k_["bidx"] == b)
            prior[rows] = [labels[r] in seen for r in rows]
            seen |= {labels[r] for r in rows if k_["admit"][r] and io[r]}
        recurrence[ds] = {}
        for flag in (False, True):
            sel = io & (prior == flag)
            recurrence[ds][str(flag)] = {"n": int(sel.sum()), **{m: 100 * auroc(k_["scores"][m][~io], k_["scores"][m][sel])
                                                                  for m in ("tins", "clavism", "nomem")}}

    # SSB-hard: WordNet similarity between the OOD class and the nearest ID candidate class of the view
    ssb = {}
    try:
        import nltk
        nltk.data.path.insert(0, str(C.NLTK_DATA))
        from nltk.corpus import wordnet as wn
        wnids = sorted(p.name for p in (C.IMAGENET_ROOT / "train").iterdir() if p.is_dir())
        k_ = near_keep["ssb_hard"]
        io = k_["is_ood"]
        oo_pos = np.flatnonzero(io)
        cache = {}

        def syn(w):
            if w not in cache:
                cache[w] = wn.synset_from_pos_and_offset("n", int(w[1:]))
            return cache[w]
        sims = []
        for pos in oo_pos:
            w_ood = cls_of(k_["ids"][pos])
            w_id = wnids[cstar[k_["q"][pos]]]
            sims.append(syn(w_ood).wup_similarity(syn(w_id)) or 0.0)
        sims = np.array(sims)
        bins = [(0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
        dvis = -v["d"][k_["q"]]
        for lo, hi in bins:
            sel = oo_pos[(sims >= lo) & (sims < hi)]
            ssb[f"{lo}-{hi}"] = {"n": int(len(sel)), "visual_d": 100 * auroc(dvis[~io], dvis[sel]),
                                 **{m: 100 * auroc(k_["scores"][m][~io], k_["scores"][m][sel]) for m in ("tins", "clavism", "nomem")}}
        ssb["median_wup"] = float(np.median(sims))
    except Exception as exc:  # noqa: BLE001
        ssb["error"] = repr(exc)
    d_q = {}
    for ds in ("ssb_hard", "ninco"):
        k_ = near_keep[ds]
        io = k_["is_ood"]
        dd = v["d"][k_["q"]]
        d_q[ds] = {"median_d_ID": float(np.median(dd[~io])), "median_d_OOD": float(np.median(dd[io])),
                   "frac_OOD_below_ID_q95": float((dd[io] < np.quantile(dd[~io], 0.95)).mean())}

    # aggregate over seeds
    def agg(ds, key, metric):
        vals = [results[(ds, s)][key][metric] for s in SEEDS]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "seed123": vals[0]}
    keys = list(results[("ssb_hard", 123)].keys())
    table = {ds: {k: {mm: agg(ds, k, mm) for mm in ("AUROC", "FPR95")} for k in keys} for ds in OO}
    for grp, names in (("near", ["ssb_hard", "ninco"]), ("far", ["inaturalist", "textures", "openimageo"])):
        table[grp] = {}
        for k in keys:
            table[grp][k] = {}
            for mm in ("AUROC", "FPR95"):
                per_seed = [np.mean([results[(n, s)][k][mm] for n in names]) for s in SEEDS]
                table[grp][k][mm] = {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
                                     "seed123": float(per_seed[0])}
    diag_mean = {ds: {k: (float(np.mean([diag[(ds, s)][k] for s in SEEDS])) if diag[(ds, 123)][k] is not None else None)
                      for k in diag[(ds, 123)]} for ds in OO}
    conf_mean = {ds: {k: float(np.mean([conf[(ds, s)][k] for s in SEEDS])) for k in conf[(ds, 123)]} for ds in OO}
    out = {"table": table, "published_tins": PUB, "permfirst_seed123": perm, "combined": combined, "delong_seed123": delong,
           "bootstrap_seed123": boot, "bootstrap_B": B, "n_ood_classes": n_cls, "diagnostics_mean": diag_mean,
           "diagnostics_seed123": {ds: diag[(ds, 123)] for ds in OO}, "conformal_static_ID": conf_static,
           "conformal_stream_ID_mean": conf_mean, "position_quartiles_seed123": position, "recurrence_seed123": recurrence,
           "ssb_wordnet": ssb, "d_quantiles": d_q, "seconds": round(time.time() - start, 1)}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"done": True, "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
