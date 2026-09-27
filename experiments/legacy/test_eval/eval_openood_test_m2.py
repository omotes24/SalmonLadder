"""THIRD evaluation on the sealed OpenOOD v1.5 ImageNet-1K test: the frozen CLAVIS-M2 (run exactly once).

Pre-registered in test_eval/prereg_test3.json before any CLAVIS-M2 score on the test is computed.
  TINS S_final : test_runs/default/<ds>_seed<s>.npz, stream orders 123..127 (the paper's main-table runs)
  features     : B/14 test_eval/results/features.pt (first evaluation), L/14 test_eval/results/features_vitl14.pt
  CLAVIS-M c1  : recomputed on the same runs (reference); CLAVIS-M2: vins/iter2.py with iter2/frozen_m2.json
Memories are reset for every ID+OOD stream. Nothing else is evaluated.
"""
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import score_stream, views_from_arrays  # noqa: E402
from vins.iter2 import score_stream_cand  # noqa: E402
from vins.stats import auroc, delong_paired, draw_members, precompute_members  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

HERE = Path(__file__).resolve().parent
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
GROUPS = {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]}
SEEDS = [123, 124, 125, 126, 127]
R = C.WORK / "test_runs" / "default"
RES = C.WORK / "test_eval" / "results"
G = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_worker(seed):
    rng = np.random.default_rng(seed)
    idc = rng.integers(0, 1000, 1000)
    res = {}
    for ds in OO:
        st = G["streams"][ds]
        id_pos = np.concatenate([st["id_members"][c] for c in idc])
        ood_pos = draw_members(st["ood_pre"], rng)
        for m, s in st["scores"].items():
            res[(ds, m)] = auroc(s[id_pos], s[ood_pos])
    return res


def run_one(task):
    ds, seed = task
    views, queries, row_of, cfg, t = G["views"], G["queries"], G["row_of"], G["cfg"], G["t"]

    def met(s, is_id):
        a, _, f = t.get_measures(s[is_id].astype(np.float64), s[~is_id].astype(np.float64))
        return {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}

    run = np.load(R / f"{ds}_seed{seed}.npz")
    ids, is_ood = run["sample_id"], run["is_ood"].astype(bool)
    is_id = ~is_ood
    s, bidx = run["S_final"].astype(np.float64), run["batch_index"]
    q = np.array([row_of[x] for x in ids])
    vB = views["B14"]
    p0, _, adm0 = score_stream(vB, queries["B14"][q], vB["d"][q], vB["p_all"][q], bidx, 0.10, 2, "diff")
    prod, sc, mem = np.ones(len(s)), {"tins": s, "clavism_c1": s * p0}, {"c1_ID_admitted": float(adm0[is_id].mean())}
    for name in ("B14", "L14"):
        v = views[name]
        p_t, g, admit, cmask = score_stream_cand(v, queries[name][q], v["d"][q], v["p_all"][q], bidx,
                                                 eps_cand=cfg["eps_cand"], eps_admit=cfg["eps_admit"],
                                                 m_mem=cfg["m_mem"], kind=cfg["kind"])
        prod *= p_t
        sc[f"S*pt_{name}"] = s * p_t
        mem[name] = {"ID_admitted": float(admit[is_id].mean()), "OOD_admitted": float(admit[is_ood].mean()),
                     "ID_candidates": float(cmask[is_id].mean()), "OOD_candidates": float(cmask[is_ood].mean())}
    sc["clavism_m2"] = s * prod
    sc["visual_prod"] = prod
    res = {k: met(val, is_id) for k, val in sc.items()}
    kp = {"ids": ids, "is_ood": is_ood, "scores": {m: sc[m] for m in ("tins", "clavism_c1", "clavism_m2")}} if seed == 123 else None
    return ds, seed, res, mem, kp


def main():
    start = time.time()
    out = C.WORK / "test_eval" / "results_m2"
    if (out / "metrics.json").exists():
        raise SystemExit("already evaluated once; refusing to run again")
    prereg = json.loads((HERE / "prereg_test3.json").read_text())
    prereg_sha = sha256(HERE / "prereg_test3.json")
    frozen_path = C.WORK / "iter2" / "frozen_m2.json"
    assert sha256(frozen_path) == prereg["frozen_candidate"]["sha256"], "frozen candidate changed"
    looks = [json.loads(x) for x in (C.WORK / "iter2" / "dev2_looks.jsonl").read_text().splitlines() if x.strip()]
    mine = [r for r in looks if r["frozen_sha256"] == prereg["frozen_candidate"]["sha256"]]
    assert mine and mine[0]["verdict"]["pass"] and len(looks) == 1, "dev2 confirmation missing or not the first look"
    l14_sha = (RES / "features_vitl14.sha256").read_text().strip()
    assert sha256(RES / "features_vitl14.pt") == l14_sha, "L/14 features changed after extraction"
    cfg = json.loads(frozen_path.read_text())["config"]
    assert cfg["dino_files"] == ["dino.pt", "dino_vitl14.pt"]
    out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(RES / "features.pt", map_location="cpu")
    blob_l = torch.load(RES / "features_vitl14.pt", map_location="cpu")
    assert blob_l["paths"] == blob["paths"]
    n_sup, n_cal = 1000 * C.N_SUPPORT, 1000 * C.N_CALIB
    cand = np.asarray(blob["cand"])
    views, queries = {}, {}
    for name, arr in (("B14", blob["dino"]), ("L14", blob_l["dino"])):
        feats = arr.numpy().astype(np.float32)
        support = feats[:n_sup].reshape(1000, C.N_SUPPORT, -1)
        queries[name] = feats[n_sup:]
        is_cal = np.zeros(len(queries[name]), dtype=bool)
        is_cal[:n_cal] = True
        views[name] = views_from_arrays(support, queries[name], cand, is_cal, m=cfg["m"])
    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    test_ids = sorted({x for ds in OO for x in np.load(R / f"{ds}_seed123.npz")["sample_id"].tolist()})
    assert len(test_ids) == len(queries["B14"]) - n_cal
    row_of = {s: n_cal + i for i, s in enumerate(test_ids)}
    t = import_tins()

    G.update({"views": views, "queries": queries, "row_of": row_of, "cfg": cfg, "t": t})
    with mp.get_context("fork").Pool(8) as pool:
        outs = pool.map(run_one, [(ds, seed) for ds in OO for seed in SEEDS])
    results, memory, keep = {}, {}, {}
    for (ds, seed, res, mem, kp) in outs:
        results[(ds, seed)] = res
        memory[f"{ds}_seed{seed}"] = mem
        if kp is not None:
            keep[ds] = kp
        print(json.dumps({"ds": ds, "seed": seed, **{m: round(res[m]["AUROC"], 2) for m in ("tins", "clavism_c1", "clavism_m2")}}))

    methods = list(results[(OO[0], SEEDS[0])].keys())
    table = {}
    for ds in OO:
        table[ds] = {m: {k: {"mean": float(np.mean([results[(ds, s_)][m][k] for s_ in SEEDS])),
                             "sd": float(np.std([results[(ds, s_)][m][k] for s_ in SEEDS], ddof=1))}
                         for k in ("AUROC", "FPR95")} for m in methods}
    for grp, names in GROUPS.items():
        table[grp] = {m: {k: {"mean": float(np.mean([np.mean([results[(n, s_)][m][k] for n in names]) for s_ in SEEDS])),
                              "sd": float(np.std([np.mean([results[(n, s_)][m][k] for n in names]) for s_ in SEEDS], ddof=1))}
                          for k in ("AUROC", "FPR95")} for m in methods}

    delong = {}
    for ds in OO:
        io, sc = keep[ds]["is_ood"], keep[ds]["scores"]
        for a_, b_ in (("clavism_m2", "clavism_c1"), ("clavism_m2", "tins")):
            aa, bb, z, pv = delong_paired(sc[a_][~io], sc[a_][io], sc[b_][~io], sc[b_][io])
            delong[f"{ds}|{a_}-{b_}"] = {"auc_a": 100 * aa, "auc_b": 100 * bb, "z": z, "p": pv}

    def cls_of(sid):
        rel = manifest[sid]["relative_path"]
        if sid.startswith("imagenet"):
            return f"id{manifest[sid]['label']}"
        parts = rel.split("/")
        return parts[1] if len(parts) > 2 and not parts[1] == "images" else sid

    G["streams"] = {}
    for ds in OO:
        io = keep[ds]["is_ood"]
        labels = np.array([cls_of(x) for x in keep[ds]["ids"]])
        id_pos = np.flatnonzero(~io)
        id_lab = np.array([int(labels[i][2:]) for i in id_pos])
        ood_pos = np.flatnonzero(io)
        order, starts, counts = precompute_members(labels[ood_pos])
        G["streams"][ds] = {"id_members": [id_pos[id_lab == c] for c in range(1000)],
                            "ood_pre": (ood_pos[order], starts, counts), "scores": keep[ds]["scores"]}
    with mp.get_context("fork").Pool(16) as pool:
        reps = pool.map(boot_worker, range(1000))
    boot = {}
    for a_, b_ in (("clavism_m2", "clavism_c1"), ("clavism_m2", "tins")):
        for ds in OO:
            arr = np.array([r[(ds, a_)] - r[(ds, b_)] for r in reps]) * 100
            boot[f"{ds}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
        for grp, names in GROUPS.items():
            arr = np.array([np.mean([r[(n, a_)] - r[(n, b_)] for n in names]) for r in reps]) * 100
            boot[f"{grp}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
    diff123 = {}
    for a_, b_ in (("clavism_m2", "clavism_c1"), ("clavism_m2", "tins")):
        for ds in OO:
            diff123[f"{ds}|{a_}-{b_}"] = results[(ds, 123)][a_]["AUROC"] - results[(ds, 123)][b_]["AUROC"]
        for grp, names in GROUPS.items():
            diff123[f"{grp}|{a_}-{b_}"] = float(np.mean([results[(n, 123)][a_]["AUROC"] - results[(n, 123)][b_]["AUROC"]
                                                         for n in names]))
    adm = {name: float(np.mean([memory[k][name]["ID_admitted"] for k in memory])) for name in ("B14", "L14")}
    blob_out = {"prereg_sha256": prereg_sha, "frozen": cfg, "l14_features_sha256": l14_sha,
                "per_run": {f"{k[0]}|{k[1]}": v for k, v in results.items()}, "table": table, "delong_seed123": delong,
                "bootstrap_seed123": boot, "diff_seed123": diff123, "memory": memory, "mean_ID_admission": adm,
                "seconds": round(time.time() - start, 1)}
    (out / "metrics.json").write_text(json.dumps(blob_out, indent=1) + "\n")

    labels = {"tins": "TINS", "clavism_c1": "CLAVIS-M", "clavism_m2": "CLAVIS-M2", "S*pt_B14": "S×p_t(B/14)",
              "S*pt_L14": "S×p_t(L/14)", "visual_prod": "視覚pの積"}
    L = ["# OpenOOD ImageNet-1K test：CLAVIS-M2（test の 3 回目の使用、1 回だけの評価）", "",
         f"- 事前登録: `test_eval/prereg_test3.json`（sha256 `{prereg_sha[:16]}…`）、凍結候補 `iter2/frozen_m2.json`。",
         "- TINS は upstream 無改変（Codex の設定）の 5 つのストリーム順序（123〜127）。値は 5 順序の平均（AUROC ± SD / FPR95、%）。", "",
         "| データセット | " + " | ".join(labels[m] for m in methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for key in OO + ["near", "far"]:
        cells = [f"{table[key][m]['AUROC']['mean']:.2f} ± {table[key][m]['AUROC']['sd']:.2f} / {table[key][m]['FPR95']['mean']:.2f}"
                 for m in methods]
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    L += ["", "| 差（シード123） | AUROC 差 | 95% 区間（クラス単位ブートストラップ） | DeLong p |", "|---|---|---|---|"]
    for a_, b_ in (("clavism_m2", "clavism_c1"), ("clavism_m2", "tins")):
        for key in OO + ["near", "far"]:
            ci = boot[f"{key}|{a_}-{b_}"]
            dl = delong.get(f"{key}|{a_}-{b_}", {}).get("p")
            L.append(f"| {labels[a_]} − {labels[b_]}：{key} | {diff123[f'{key}|{a_}-{b_}']:+.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                     + (f"{dl:.2e}" if dl is not None else "–") + " |")
    L += ["", f"- メモリに入った ID の割合（平均）：B/14 {100 * adm['B14']:.2f}%、L/14 {100 * adm['L14']:.2f}%（目安 10%）"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
