"""FOURTH evaluation on the sealed OpenOOD v1.5 ImageNet-1K test: the frozen CLAVIS-M3 (run exactly once).

Pre-registered in test_eval/prereg_test4.json before any CLAVIS-M3 score on the test is computed.
  TINS S_final : test_runs/default/<ds>_seed<s>.npz, stream orders 123..127
  features     : B/14 test_eval/results/features.pt, L/14 test_eval/results/features_vitl14.pt (stored earlier)
  CLAVIS-M3    : functions of scripts/iter3_eval.py (sha256 checked); CLAVIS-M2 recomputed as the reference
Memories and graphs are reset for every ID+OOD stream. Nothing else is evaluated.
"""
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import views_from_arrays  # noqa: E402
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


def load_ev():
    spec = importlib.util.spec_from_file_location("iter3_eval", ROOT / "scripts" / "iter3_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def run_one(task):
    ds, seed = task
    ev, t, cfg = G["ev"], G["t"], G["cfg"]
    k_, a_, g_ = cfg["lp"].split(",")

    def met(s, is_id):
        a, _, f = t.get_measures(s[is_id].astype(np.float64), s[~is_id].astype(np.float64))
        return {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}

    run = np.load(R / f"{ds}_seed{seed}.npz")
    ids, is_ood = run["sample_id"], run["is_ood"].astype(bool)
    is_id = ~is_ood
    s, bidx = run["S_final"].astype(np.float64), run["batch_index"]
    q = np.array([G["row_of"][x] for x in ids])
    m2, m3_mem, m3_lp, mem = s.copy(), np.ones(len(s)), np.ones(len(s)), {}
    for name in ("B14", "L14"):
        vk, vp, Q = G["knn"][name], G["proto"][name], G["queries"][name]
        sf = Q[q]
        p2, _, _, _ = score_stream_cand(vk, sf, vk["d"][q], vk["p_all"][q], bidx, eps_cand=0.30, eps_admit=0.10)
        m2 *= p2
        admit = ev.admit_mask(vp, sf, vp["d"][q], vp["p_all"][q], bidx, cfg["rule"])
        p_t, _ = ev.p_memory(vp, sf, vp["d"][q], bidx, admit)
        p_lp = ev.lp_pvalues(vp, sf, bidx, int(k_), float(a_), float(g_))
        m3_mem *= p_t
        m3_lp *= p_lp
        mem[name] = {"ID_admitted": float(admit[is_id].mean()), "OOD_admitted": float(admit[is_ood].mean())}
    sc = {"tins": s, "clavism_m2": m2, "clavism_m3": s * m3_mem * m3_lp, "S*memory": s * m3_mem, "S*lp": s * m3_lp}
    res = {k: met(v, is_id) for k, v in sc.items()}
    kp = {"ids": ids, "is_ood": is_ood, "scores": {m: sc[m] for m in ("tins", "clavism_m2", "clavism_m3")}} if seed == 123 else None
    return ds, seed, res, mem, kp


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


def main():
    start = time.time()
    out = C.WORK / "test_eval" / "results_m3"
    if (out / "metrics.json").exists():
        raise SystemExit("already evaluated once; refusing to run again")
    prereg = json.loads((HERE / "prereg_test4.json").read_text())
    prereg_sha = sha256(HERE / "prereg_test4.json")
    frozen_path = C.WORK / "iter3" / "frozen_m3.json"
    assert sha256(frozen_path) == prereg["frozen_candidate"]["sha256"], "frozen candidate changed"
    assert sha256(ROOT / "scripts" / "iter3_eval.py") == prereg["code"]["sha256"], "code changed"
    looks = [json.loads(x) for x in (C.WORK / "iter3" / "dev2_looks.jsonl").read_text().splitlines() if x.strip()]
    assert looks and looks[0]["frozen_sha256"] == prereg["frozen_candidate"]["sha256"] and looks[0]["verdict"]["pass"], \
        "first dev2 look missing or not passed"
    assert sha256(RES / "features_vitl14.pt") == (RES / "features_vitl14.sha256").read_text().strip()
    cfg = json.loads(frozen_path.read_text())["config"]
    assert cfg["views"] == ["dino.pt:proto", "dino_vitl14.pt:proto"] and cfg["combo"] == "0,1,2,3"
    out.mkdir(parents=True, exist_ok=True)
    ev = load_ev()

    blob = torch.load(RES / "features.pt", map_location="cpu")
    blob_l = torch.load(RES / "features_vitl14.pt", map_location="cpu")
    assert blob_l["paths"] == blob["paths"]
    paths = blob["paths"]
    n_sup, n_cal = 1000 * C.N_SUPPORT, 1000 * C.N_CALIB
    wn = lambda p: Path(p).parent.name  # noqa: E731
    assert all(wn(paths[n_sup + 4 * c + j]) == wn(paths[12 * c]) for c in range(1000) for j in range(4)), "calibration order"
    cal_cls = np.repeat(np.arange(1000), 4)
    cand = np.asarray(blob["cand"])
    knn, proto, queries = {}, {}, {}
    for name, arr in (("B14", blob["dino"]), ("L14", blob_l["dino"])):
        feats = arr.numpy().astype(np.float32)
        support = feats[:n_sup].reshape(1000, C.N_SUPPORT, -1)
        queries[name] = feats[n_sup:]
        is_cal = np.zeros(len(queries[name]), dtype=bool)
        is_cal[:n_cal] = True
        knn[name] = views_from_arrays(support, queries[name], cand, is_cal, m=2)
        vp = ev.custom_view(support, queries[name], cand, is_cal, cal_cls, m=2, proto=True)
        vp["support_arr"] = vp["support"]
        proto[name] = vp
    test_ids = sorted({x for ds in OO for x in np.load(R / f"{ds}_seed123.npz")["sample_id"].tolist()})
    assert len(test_ids) == len(queries["B14"]) - n_cal
    row_of = {s: n_cal + i for i, s in enumerate(test_ids)}
    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    t = import_tins()
    G.update({"ev": ev, "t": t, "cfg": cfg, "knn": knn, "proto": proto, "queries": queries, "row_of": row_of})
    tasks = [(ds, seed) for ds in ("ssb_hard", "openimageo", "inaturalist", "ninco", "textures") for seed in SEEDS]
    with mp.get_context("fork").Pool(8) as pool:
        outs = pool.map(run_one, tasks, chunksize=1)
    results, memory, keep = {}, {}, {}
    for ds, seed, res, mem, kp in outs:
        results[(ds, seed)] = res
        memory[f"{ds}_seed{seed}"] = mem
        if kp is not None:
            keep[ds] = kp
        print(json.dumps({"ds": ds, "seed": seed, **{m: round(res[m]["AUROC"], 2) for m in ("tins", "clavism_m2", "clavism_m3")}}))

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
        for a_, b_ in (("clavism_m3", "clavism_m2"), ("clavism_m3", "tins")):
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
    boot, diff123 = {}, {}
    for a_, b_ in (("clavism_m3", "clavism_m2"), ("clavism_m3", "tins")):
        for ds in OO:
            arr = np.array([r[(ds, a_)] - r[(ds, b_)] for r in reps]) * 100
            boot[f"{ds}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
            diff123[f"{ds}|{a_}-{b_}"] = results[(ds, 123)][a_]["AUROC"] - results[(ds, 123)][b_]["AUROC"]
        for grp, names in GROUPS.items():
            arr = np.array([np.mean([r[(n, a_)] - r[(n, b_)] for n in names]) for r in reps]) * 100
            boot[f"{grp}|{a_}-{b_}"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float((arr <= 0).mean())]
            diff123[f"{grp}|{a_}-{b_}"] = float(np.mean([results[(n, 123)][a_]["AUROC"] - results[(n, 123)][b_]["AUROC"] for n in names]))
    adm = {name: float(np.mean([memory[k][name]["ID_admitted"] for k in memory])) for name in ("B14", "L14")}
    blob_out = {"prereg_sha256": prereg_sha, "frozen": cfg, "per_run": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
                "table": table, "delong_seed123": delong, "bootstrap_seed123": boot, "diff_seed123": diff123,
                "memory": memory, "mean_ID_admission": adm, "seconds": round(time.time() - start, 1)}
    (out / "metrics.json").write_text(json.dumps(blob_out, indent=1) + "\n")
    labels = {"tins": "TINS", "clavism_m2": "CLAVIS-M2", "clavism_m3": "CLAVIS-M3", "S*memory": "S×メモリp", "S*lp": "S×LP p"}
    L = ["# OpenOOD ImageNet-1K test：CLAVIS-M3（test の 4 回目の使用、1 回だけの評価）", "",
         f"- 事前登録: `test_eval/prereg_test4.json`（sha256 `{prereg_sha[:16]}…`）、凍結候補 `iter3/frozen_m3.json`。",
         "- 5 つのストリーム順序（123〜127）の平均（AUROC ± SD / FPR95、%）。", "",
         "| データセット | " + " | ".join(labels[m] for m in methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for key in OO + ["near", "far"]:
        L.append(f"| {key} | " + " | ".join(
            f"{table[key][m]['AUROC']['mean']:.2f} ± {table[key][m]['AUROC']['sd']:.2f} / {table[key][m]['FPR95']['mean']:.2f}"
            for m in methods) + " |")
    L += ["", "| 差（シード123） | AUROC 差 | 95% 区間 | DeLong p |", "|---|---|---|---|"]
    for a_, b_ in (("clavism_m3", "clavism_m2"), ("clavism_m3", "tins")):
        for key in OO + ["near", "far"]:
            ci = boot[f"{key}|{a_}-{b_}"]
            dl = delong.get(f"{key}|{a_}-{b_}", {}).get("p")
            L.append(f"| {labels[a_]} − {labels[b_]}：{key} | {diff123[f'{key}|{a_}-{b_}']:+.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                     + (f"{dl:.2e}" if dl is not None else "–") + " |")
    L += ["", f"- メモリに入った ID の割合（平均）：B/14 {100 * adm['B14']:.2f}%、L/14 {100 * adm['L14']:.2f}%"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
