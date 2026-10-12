"""Phase 18 runner (descriptive; see p18common). Steps:
  --step main   : OpenOOD v1.5 ImageNet-1K (5 OOD sets) and Four-OOD (4), six draws, views L14 + D3L: (a) read-outs read with R,
                  per view and summed; (d) sensitivity (k_g, lambda on the same graph; 4 / 8 support images per class on a rebuilt
                  graph); p_fin of p+ and sigma_ref with validity. p+ and sigma_ref are checked against Phase 17 per task.
  --step views  : (b) CLIP ViT-B/16 and DINO ViT-B/16 alone, draw p3, the read-outs of (a).
  --step valrc  : (c) R and C from val_imagenet.txt (4 per class), six draws, p+ and sigma_ref with validity and seed ID fraction.
  --step timing : (e) the method (graph, u+, seeds, u-, read-out) on OpenOOD SSB-hard, draw p3, TIMING_REPS times.
Output: P18/results/p18_<step>/<part>/<draw>/<ds>.{csv,meta.json} (timing: p18_timing/timing.json)."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p18common as P
import core14 as K
import core18 as C18
from draws12 import pool_shots, datasets, DRAWS

VIEWS = list(P.VIEWS18)
PARTS = ("openood", "fourood")
TINS_ONLY = True


def shots_of(D, d):
    if d == "p3":
        return {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    return pool_shots(d)


def tins_bases(logS):
    return {("TINS", int(seed)): S for seed, S in logS.items()}


def save(out_dir, name, rows, meta, t0):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta["seconds"] = round(time.time() - t0, 1)
    meta["utc"] = P.utc()
    pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(json.dumps({"task": str(out_dir.relative_to(P.RESULTS18) / name), "seconds": meta["seconds"], "utc": meta["utc"],
                      "check17": meta.get("check17")}), flush=True)


def seed_cols(infos, prefix=""):
    out = {}
    for v, e in infos.items():
        out[f"{prefix}seeds_{v}"] = e["seeds"]
        out[f"{prefix}cal_seeds_{v}"] = e["cal_seeds"]
        out[f"{prefix}idfrac_{v}"] = e["idfrac"]
        out[f"{prefix}pi0_{v}"] = e["pi0_hat"]
    return out


def check17(part, d, ds, rows):
    """|difference| of p+ / sigma_ref (AUROC, FPR95 standalone; false-alarm rate of p_fin at 0.05) against Phase 17 (same task)."""
    f = P.RESULTS17 / "p17_test" / part / d / f"{ds}.csv"
    r17 = pd.read_csv(f)
    r18 = pd.DataFrame(rows)
    pairs = [("lp", "ref|lp|stat", ("AUROC", "FPR95")), ("sref", "ref|storey|q0.2|pp|stat", ("AUROC", "FPR95")),
             ("lp|pfin", "ref|lp", ("AUROC", "FPR95", "fpr@0.05", "bh_fdp")), ("sref|pfin", "ref|storey|q0.2|pp", ("AUROC", "FPR95", "fpr@0.05", "bh_fdp"))]
    diff = 0.0
    for a, b, cols in pairs:
        x = r18[(r18.cfg == a) & (r18.base == "none")].iloc[0]
        y = r17[(r17.cfg == b) & (r17.base == "none")].iloc[0]
        for c in cols:
            diff = max(diff, abs(float(x[c]) - float(y[c])))
    tins = r18[(r18.cfg == "lp") & (r18.base == "TINS")].set_index("seed").FPR95
    tins17 = r17[(r17.cfg == "ref|lp|stat") & (r17.base == "TINS")].set_index("seed").FPR95
    diff = max(diff, float((tins - tins17.loc[tins.index]).abs().max()))
    return diff


def run_main(worker, nworkers, device, only=None):
    labels_sup = np.repeat(np.arange(1000), 12)
    for part in PARTS:
        tasks_all = [(d, ds) for d in DRAWS for ds in sorted(datasets(part))]
        tasks = [t for t in tasks_all if only is None or (part, t[0], t[1]) == only][worker::nworkers]
        out_root = P.RESULTS18 / "p18_main" / part
        todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
        if not todo:
            continue
        D = P.load_p3(part)
        P.add_views(D, part, ["D3L"])
        sets = datasets(part)
        shots = {d: shots_of(D, d) for d in sorted(set(d for d, _ in todo))}
        for d, ds in todo:
            t0 = time.time()
            ids, flag, logS = sets[ds]
            rows_ev = np.array([D["row"][x] for x in ids])
            bases = tins_bases(logS)
            tag = {"step": "main", "part": part, "draw": d, "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum())}
            tm = C18.Timer()
            st, info, nC, nnodes = {}, defaultdict(dict), None, {}
            for v in VIEWS:
                sup, cal = shots[d][v]
                sf = P.l2n(D["ev"][v][rows_ev - 4000])
                with tm(f"graph_{v}"):
                    g = K.Graph(sup, cal, sf, device)
                nnodes[v] = int(g.N)
                R, C, T, pool = C18.nodes(g)
                nC = len(C)
                lp, ln, inf, _ = C18.lp_sref(g, R, C, pool, flag, P.KG18, P.LAM18, tm, f"_{v}")
                st[("lp", v)], st[("ln", v)], info["main"][v] = lp, ln, inf
                for k_, x in C18.baselines(g, sup, cal, sf, R, pool, labels_sup, 1000, device, 12, tm).items():
                    st[(k_, v)] = x
                for name, (k, lam) in P.SENS18.items():
                    lp2, ln2, inf2, _ = C18.lp_sref(g, R, C, pool, flag, k, lam)
                    st[(f"lp|{name}", v)], st[(f"ln|{name}", v)], info[name][v] = lp2, ln2, inf2
                g.free()
                for s in P.SHOTS18:
                    sup_s = np.ascontiguousarray(sup.reshape(1000, 12, -1)[:, :s].reshape(-1, sup.shape[1]))
                    g = K.Graph(sup_s, cal, sf, device)
                    R2, C2, T2, pool2 = C18.nodes(g)
                    assert len(C2) == nC and g.nt == len(ids)
                    lp2, ln2, inf2, _ = C18.lp_sref(g, R2, C2, pool2, flag, P.KG18, P.LAM18)
                    st[(f"lp|shot{s}", v)], st[(f"ln|shot{s}", v)], info[f"shot{s}"][v] = lp2, ln2, inf2
                    b2 = C18.baselines(g, sup_s, cal, sf, R2, pool2, None, 1000, device, s, with_lgc=False)
                    st[(f"static|shot{s}", v)], st[(f"knn|shot{s}", v)] = b2["static"], b2["knn"]
                    g.free()

            def two(name):
                return sum(st[(name, v)] for v in VIEWS)
            rows = []
            for name in ("lp", "onestep", "static", "knn", "lgc"):
                rows += C18.eval_rows(tag, name, two(name), nC, flag, bases)
                for v in VIEWS:
                    rows += C18.eval_rows(tag, f"{name}|{v}", st[(name, v)], nC, flag, bases)
            rows += C18.eval_rows(tag, "sref", two("lp") + two("ln"), nC, flag, bases, seed_cols(info["main"]))
            for v in VIEWS:
                rows += C18.eval_rows(tag, f"sref|{v}", st[("lp", v)] + st[("ln", v)], nC, flag, bases)
            for name in list(P.SENS18) + [f"shot{s}" for s in P.SHOTS18]:
                rows += C18.eval_rows(tag, f"lp|{name}", two(f"lp|{name}"), nC, flag, bases)
                rows += C18.eval_rows(tag, f"sref|{name}", two(f"lp|{name}") + two(f"ln|{name}"), nC, flag, bases, seed_cols(info[name]))
            for s in P.SHOTS18:
                rows += C18.eval_rows(tag, f"static|shot{s}", two(f"static|shot{s}"), nC, flag, bases)
                rows += C18.eval_rows(tag, f"knn|shot{s}", two(f"knn|shot{s}"), nC, flag, bases)
            r, _ = C18.pfin_rows(tag, "lp|pfin", two("lp"), nC, flag, bases)
            rows += r
            r, _ = C18.pfin_rows(tag, "sref|pfin", two("lp") + two("ln"), nC, flag, bases)
            r[0].update(seed_cols(info["main"]))                       # r[0] = the standalone row of sref|pfin
            rows += r
            meta = {"nodes": nnodes, "seconds_by_part": tm.t, "seeds": info}
            meta["check17"] = check17(part, d, ds, rows)
            assert meta["check17"] < 1e-4, ("Phase 17 mismatch", part, d, ds, meta["check17"])
            save(out_root / d, ds, rows, meta, t0)


def run_views(worker, nworkers, device, only=None):
    labels_sup = np.repeat(np.arange(1000), 12)
    for part in PARTS:
        tasks_all = [(bb, ds) for bb in P.BACKBONES18 for ds in sorted(datasets(part))]
        tasks = [t for t in tasks_all if only is None or (part, t[0], t[1]) == only][worker::nworkers]
        out_root = P.RESULTS18 / "p18_views" / part
        todo = [(bb, ds) for bb, ds in tasks if not (out_root / bb / f"{ds}.csv").exists()]
        if not todo:
            continue
        D = P.load_p3(part)
        sets = datasets(part)
        for bb, ds in todo:
            t0 = time.time()
            ids, flag, logS = sets[ds]
            rows_ev = np.array([D["row"][x] for x in ids])
            bases = tins_bases(logS)
            tag = {"step": "views", "part": part, "draw": "p3", "ds": ds, "backbone": bb, "n": int(len(ids)), "n_ood": int(flag.sum())}
            tm = C18.Timer()
            sup, cal = P.l2n(D["shots"][bb][:12000]), P.l2n(D["shots"][bb][12000:16000])
            sf = P.l2n(D["ev"][bb][rows_ev - 4000])
            g = K.Graph(sup, cal, sf, device)
            R, C, T, pool = C18.nodes(g)
            nC = len(C)
            lp, ln, inf, _ = C18.lp_sref(g, R, C, pool, flag, P.KG18, P.LAM18, tm)
            b = C18.baselines(g, sup, cal, sf, R, pool, labels_sup, 1000, device, 12, tm)
            g.free()
            rows = C18.eval_rows(tag, f"lp|{bb}", lp, nC, flag, bases)
            rows += C18.eval_rows(tag, f"sref|{bb}", lp + ln, nC, flag, bases, seed_cols({bb: inf}))
            for k_, x in b.items():
                rows += C18.eval_rows(tag, f"{k_}|{bb}", x, nC, flag, bases)
            r, _ = C18.pfin_rows(tag, f"lp|{bb}|pfin", lp, nC, flag, bases)
            rows += r
            r, _ = C18.pfin_rows(tag, f"sref|{bb}|pfin", lp + ln, nC, flag, bases)
            rows += r
            save(out_root / bb, ds, rows, {"seeds": inf, "seconds_by_part": tm.t, "dim": int(sup.shape[1])}, t0)


def val_choice(di, Df):
    """Fourood rows (in_val part) of 4 validation images per class in class-major order [R, R, C, C], and their labels."""
    from vins import config as VC
    lines = [ln.split() for ln in (Path(VC.OPENOOD_IMGLIST_DIR) / "val_imagenet.txt").read_text().splitlines() if ln.strip()]
    test_names = {Path(ln.split()[0]).name for ln in (Path(VC.OPENOOD_IMGLIST_DIR) / "test_imagenet.txt").read_text().splitlines() if ln.strip()}
    by_cls = defaultdict(list)
    for rel, lab in lines:
        by_cls[int(lab)].append(Path(rel).name)
    pf = P.image_paths("fourood")
    pos = {Path(p).name: i for i, p in enumerate(pf[:50000])}
    rows, labs, names = [], [], []
    for c in range(1000):
        names_c = sorted(by_cls[c])
        assert len(names_c) == 5
        pick = np.random.default_rng([P.VAL_RNG, di, c]).permutation(len(names_c))[:4]
        for j in pick:
            rows.append(pos[names_c[j]])
            labs.append(c)
            names.append(names_c[j])
    assert not (set(names) & test_names)
    return np.array(rows), np.array(labs), names


def run_valrc(worker, nworkers, device, only=None):
    tasks_all = [(part, d, ds) for part in PARTS for d in DRAWS for ds in sorted(datasets(part))]
    tasks = [t for t in tasks_all if only is None or t == only][worker::nworkers]
    todo = [t for t in tasks if not (P.RESULTS18 / "p18_valrc" / t[0] / t[1] / f"{t[2]}.csv").exists()]
    if not todo:
        return
    Df = P.load_p3("fourood")
    P.add_views(Df, "fourood", ["D3L"])
    Do = None
    if any(t[0] == "openood" for t in todo):
        Do = P.load_p3("openood")
        P.add_views(Do, "openood", ["D3L"])
    choice = {d: val_choice(DRAWS.index(d), Df) for d in sorted(set(t[1] for t in todo))}
    sets = {part: datasets(part) for part in sorted(set(t[0] for t in todo))}
    for part, d, ds in todo:
        t0 = time.time()
        D = Do if part == "openood" else Df
        vrows, vlabs, vnames = choice[d]
        ids, flag, logS = sets[part][ds]
        if part == "fourood":
            drop = {Df["ids"][r] for r in vrows}
            keep = np.array([x not in drop for x in ids])
            assert int((~keep).sum()) == len(vrows), (int((~keep).sum()), len(vrows))
            ids, flag, logS = ids[keep], flag[keep], {s: S[keep] for s, S in logS.items()}
        rows_ev = np.array([D["row"][x] for x in ids])
        bases = tins_bases(logS)
        tag = {"step": "valrc", "part": part, "draw": d, "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum())}
        sup_all = shots_of(D, d)
        st, info, acc = {}, {}, {}
        nC = None
        for v in VIEWS:
            sup = sup_all[v][0]
            cal = P.l2n(Df["ev"][v][vrows])
            proto = sup.reshape(1000, 12, -1).mean(1)
            proto /= np.linalg.norm(proto, axis=1, keepdims=True)
            acc[v] = float(np.mean(np.argmax(cal @ proto.T, axis=1) == vlabs))      # labels of the list agree with the support classes
            sf = P.l2n(D["ev"][v][rows_ev - 4000])
            g = K.Graph(sup, cal, sf, device)
            R, C, T, pool = C18.nodes(g)
            nC = len(C)
            lp, ln, inf, _ = C18.lp_sref(g, R, C, pool, flag, P.KG18, P.LAM18)
            st[("lp", v)], st[("ln", v)], info[v] = lp, ln, inf
            g.free()
        lp2 = sum(st[("lp", v)] for v in VIEWS)
        ln2 = sum(st[("ln", v)] for v in VIEWS)
        rows = C18.eval_rows(tag, "lp", lp2, nC, flag, bases)
        rows += C18.eval_rows(tag, "sref", lp2 + ln2, nC, flag, bases, seed_cols(info))
        r, _ = C18.pfin_rows(tag, "lp|pfin", lp2, nC, flag, bases)
        rows += r
        r, _ = C18.pfin_rows(tag, "sref|pfin", lp2 + ln2, nC, flag, bases)
        r[0].update(seed_cols(info))
        rows += r
        save(P.RESULTS18 / "p18_valrc" / part / d, ds, rows, {"seeds": info, "proto_acc_of_val_RC": acc, "n_removed_from_test": (len(vrows) if part == "fourood" else 0),
                                                            "val_first_names": vnames[:8]}, t0)


def run_timing(device):
    out = P.RESULTS18 / "p18_timing" / "timing.json"
    if out.exists():
        return
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    sets = datasets("openood")
    ids, flag, logS = sets["ssb_hard"]
    rows_ev = np.array([D["row"][x] for x in ids])
    shots = shots_of(D, "p3")
    reps = []
    for rep in range(P.TIMING_REPS):
        torch.cuda.reset_peak_memory_stats()
        tm = C18.Timer()
        t0 = time.time()
        stat, N, nnz = 0.0, {}, {}
        for v in VIEWS:
            sup, cal = shots[v]
            sf = P.l2n(D["ev"][v][rows_ev - 4000])
            with tm("graph"):
                g = K.Graph(sup, cal, sf, device)
                M = g.matrix()
            N[v], nnz[v] = int(g.N), int(M.values().numel())
            R, C, T, pool = C18.nodes(g)
            lp, ln, inf, _ = C18.lp_sref(g, R, C, pool, flag, P.KG18, P.LAM18, tm)
            stat = stat + lp + ln
            g.free()
        with tm("pfin"):
            pfin = C18.p_low(stat[:len(C)], stat[len(C):])
        reps.append({"seconds": {k: round(x, 3) for k, x in tm.t.items()}, "total": round(time.time() - t0, 3), "nodes": N, "nnz": nnz,
                     "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
        print(json.dumps(reps[-1]), flush=True)
    P.write_json(out, {"task": "openood ssb_hard draw p3, views L14 + D3L", "gpu": torch.cuda.get_device_name(0), "reps": reps, "utc": P.utc()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["main", "views", "valrc", "timing"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--only", default=None, help="part:draw:ds (main, valrc) or part:backbone:ds (views)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    only = tuple(a.only.split(":")) if a.only else None
    if a.step == "main":
        run_main(a.worker, a.nworkers, "cuda", only)
    elif a.step == "views":
        run_views(a.worker, a.nworkers, "cuda", only)
    elif a.step == "valrc":
        run_valrc(a.worker, a.nworkers, "cuda", only)
    else:
        run_timing("cuda")
    print("P18_DONE", flush=True)
