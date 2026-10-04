"""Exp 5/6 (descriptive, after the confirmatory evaluation; no selection): batch size, fixed thresholds and
arrival conditions on U1 with the frozen REPRISE, the best minimal configuration and static x p_LP (s0 = 1 and MCM).

exp5: U1 split 1, draw 0, orders 123-125; batch sizes {1,16,64,256}.
exp6: U1 splits 1-5, draw 0, order 123; conditions random / id_first9000 / id_burst10 / ood1pct / ood5pct /
      burst10_ood5pct.
Fixed thresholds: dev1 ID scores (near streams, draws 0-4, order 123) at nominal false-alarm {0.1, 1, 5}%.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bank_data import Features, pos_text, shots_of, spec
from common import RESULTS, V5, utc
from engine import run_view
from evaluate import load_lock, parse
from metrics_p4 import metrics
from static import from_proto, static_view

OUT = RESULTS / "operating"
REF = "k10g1l0.9"
NOMINAL = (0.001, 0.01, 0.05)


def family_list(lock):
    best = lock["best_minimal"]
    return {"REPRISE": ("rep_L0", REF), "static_pLP": ("stat_cdf_L0", REF), "best": (best, lock["selected"][best]["config"])}


def need_configs(fams):
    graphs, lams = set(), set()
    for f, c in fams.values():
        k, g, l = parse(c)
        graphs.add((k, g))
        lams.add(l)
    return tuple(sorted(graphs)), tuple(sorted(lams))


def assemble(S, fam, cfg):
    b = {"stat_cdf": "cdf", "rep_L1": "cdf", "stat_cdf_L0": "cdf_L0", "rep_L0": "cdf_L0"}.get(fam, fam)
    x = S[f"{b}|{cfg}"]
    if fam.startswith("stat_"):
        x = S["static"] + x
    if fam.startswith("rep_"):
        x = S["Mpt"] + x
    return x


def base_family(f):
    return {"stat_cdf": "cdf", "rep_L1": "cdf", "stat_cdf_L0": "cdf_L0", "rep_L0": "cdf_L0"}.get(f, f)


def run_order(F, inp, order_ids, bidx, fams):
    graphs, lams = need_configs(fams)
    need = tuple(sorted({base_family(f) for f, _ in fams.values()} | {"cdf_L0"}))
    pos = {s: i for i, s in enumerate(inp["ids_all"])}
    ix = np.array([pos[s] for s in order_ids])
    views = {}
    t0 = time.time()
    for v in ("B14", "L14"):
        st_all = inp[f"static_{v}"]
        nc = len(inp[f"cal_{v}"])
        st = {**st_all, "d": np.r_[st_all["d"][:nc], st_all["d"][nc:][ix]], "d_all": np.r_[st_all["d_all"][:nc], st_all["d_all"][nc:][ix]],
              "p": np.r_[st_all["p"][:nc], st_all["p"][nc:][ix]]}
        views[v], diag = run_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"][ix], bidx, st, V5["thresholds"], graphs, lams,
                                  "cuda", need, trace=True)
    keys = [k for k in views["B14"] if not k.startswith("_")]
    S = {k: views["B14"][k] + views["L14"][k] for k in keys}
    out = {name: assemble(S, f, c) for name, (f, c) in fams.items()}
    out["pLP"], out["Mpt"] = S[f"cdf_L0|{REF}"], S["Mpt"]
    admits = {v: views[v]["_admit"] for v in ("B14", "L14")}
    return out, admits, time.time() - t0


def bank_inputs(F, split, draw=0):
    names, shots, ide, ood = spec("U1", split)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, draw, C)
    ids_all = ide + ood
    flag_all = np.r_[np.zeros(len(ide), bool), np.ones(len(ood), bool)]
    pt = pos_text("U1", split)
    clip_all, clip_cal = F.get("CLIP", ids_all), F.get("CLIP", cal_ids)
    cand_a = np.argsort(-(clip_all @ pt.T), axis=1)[:, :20]
    cand_c = np.argsort(-(clip_cal @ pt.T), axis=1)[:, :20]
    import torch
    mcm = torch.softmax(torch.as_tensor(clip_all @ pt.T, dtype=torch.float64), dim=1).max(1).values.numpy()
    inp = {"ids_all": ids_all, "flag_all": flag_all, "mcm_all": mcm, "ide": ide, "ood": ood}
    wn = [s.split("/")[2] for s in ide]
    inp["id_class"] = np.array(wn)
    for v in ("B14", "L14"):
        sup = F.get(v, sup_ids).reshape(C, 12, -1)
        cal = F.get(v, cal_ids)
        X = F.get(v, ids_all)
        inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"] = sup, cal, X
        inp[f"static_{v}"] = static_view(sup, cal, X, cand_c, cand_a, V5["n0"], V5["m"])
    return inp


def orders(inp, cond, seed):
    rng = np.random.default_rng([20260928, 60, seed])
    ide, ood = list(inp["ide"]), list(inp["ood"])
    if cond == "random":
        from vins.tins_dev import build_order
        o = build_order(len(ide), len(ood), seed)
        return [ide[i] if f == 0 else ood[i] for f, i in o]
    if cond.startswith("ood"):
        pct = {"ood1pct": 0.01, "ood5pct": 0.05}[cond]
        n_ood = int(round(pct * len(ide) / (1 - pct)))
        ood = [ood[i] for i in rng.permutation(len(ood))[:n_ood]]
        items = ide + ood
        return [items[i] for i in rng.permutation(len(items))]
    if cond == "id_first9000":
        p = rng.permutation(len(ide))
        head = [ide[i] for i in p[:9000]]
        rest = [ide[i] for i in p[9000:]] + ood
        return head + [rest[i] for i in rng.permutation(len(rest))]
    if cond in ("id_burst10", "burst10_ood5pct"):
        cls = np.array([s.split("/")[2] for s in ide])
        blocks = []
        for w in np.unique(cls):
            members = [ide[i] for i in np.flatnonzero(cls == w)]
            for lo in range(0, len(members), 10):
                blocks.append(members[lo:lo + 10])
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        seq = [s for b in blocks for s in b]
        if cond == "burst10_ood5pct":
            n_ood = int(round(0.05 * len(ide) / 0.95))
            ood = [ood[i] for i in rng.permutation(len(ood))[:n_ood]]
        slots = np.sort(rng.choice(len(seq) + len(ood), size=len(ood), replace=False))
        out, it_id, it_ood = [], iter(seq), iter(ood)
        sl = set(slots.tolist())
        for j in range(len(seq) + len(ood)):
            out.append(next(it_ood) if j in sl else next(it_id))
        return out
    raise ValueError(cond)


def dev1_thresholds(lock):
    """Nominal false-alarm thresholds from dev1 ID scores (near streams, draws 0-4, order 123), standalone."""
    from pilot import load_development, prepare
    fams = family_list(lock)
    graphs, lams = need_configs(fams)
    R, D = load_development()
    pool = {name: [] for name in fams}
    for draw in range(5):
        views = {}
        for v in ("B14", "L14"):
            su, ca, sf, bi, vv, sid, flag, S0 = prepare(R, D, draw, v, "near", 123)
            views[v], _ = run_view(su, ca, sf, bi, from_proto(vv, su, V5["n0"]), V5["thresholds"], graphs, lams, "cuda",
                                   ("cdf", "cdf_L0", "raw", "z", "mass", "rw", "sprop"))
        keys = [k for k in views["B14"] if not k.startswith("_")]
        S = {k: views["B14"][k] + views["L14"][k] for k in keys}
        for name, (f, c) in fams.items():
            pool[name].append(assemble(S, f, c)[~flag])
    th = {name: {str(a): float(np.quantile(np.concatenate(x), a)) for a in NOMINAL} for name, x in pool.items()}
    return th


def delays(det, flag, ids):
    """Per OOD class: 1-based count of that class's images seen until the first detection (NaN if never)."""
    cls = np.array([s.split("/")[2] for s in ids], dtype=object)
    out = []
    for w in np.unique(cls[flag]):
        idx = np.flatnonzero(flag & (cls == w))
        hit = np.flatnonzero(det[idx])
        out.append(float(hit[0] + 1) if len(hit) else np.nan)
    return np.array(out)


def operating_rows(out, flag, ids, th, extra):
    rows = []
    for name, x in out.items():
        m = metrics(x, flag)
        r = {"family": name, **m, **extra}
        if name in th:
            for a, t in th[name].items():
                acc = x >= t
                r[f"idFA@{a}"] = float(100 * np.mean(~acc[~flag]))
                r[f"oodTPR@{a}"] = float(100 * np.mean(~acc[flag]))
                dl = delays(~acc, flag, ids)
                r[f"delay_median@{a}"] = float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan
                r[f"detected_first@{a}"] = float(100 * np.mean(dl == 1))
                r[f"detected_within5@{a}"] = float(100 * np.mean(dl <= 5))
                r[f"never_detected@{a}"] = float(100 * np.mean(~np.isfinite(dl)))
        rows.append(r)
    return rows


def main_exp5(F, lock, th, seeds):
    inp = bank_inputs(F, 1)
    fams = family_list(lock)
    for seed in seeds:
        order_ids = orders(inp, "random", seed)
        oodset = set(inp["ood"])
        flag = np.array([s in oodset for s in order_ids])
        for B in (256, 64, 16, 1):
            f_out = OUT / f"exp5_B{B}_seed{seed}.parquet"
            if f_out.exists():
                continue
            bidx = np.arange(len(order_ids)) // B
            out, admits, sec = run_order(F, inp, order_ids, bidx, fams)
            rows = operating_rows(out, flag, order_ids, th, {"exp": "exp5", "batch": B, "seed": seed, "seconds": sec,
                                                             "ms_per_image": 1000 * sec / len(order_ids)})
            if B in (256, 16) and seed == seeds[0]:
                npre = 36 * 256
                pre, _, _ = run_order(F, inp, order_ids[:npre], bidx[:npre], fams)
                for r in rows:
                    if r["family"] in pre:
                        a_, b_ = pre[r["family"]], out[r["family"]][:npre]
                        r["prefix_identical"] = bool(np.array_equal(a_, b_))
                        r["prefix_max_abs_diff"] = float(np.max(np.abs(np.where(a_ == b_, 0.0, a_ - b_))))
            tmp = OUT / f"exp5_B{B}_seed{seed}.{os.getpid()}.parquet"
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, f_out)
            print(json.dumps({"exp5": f_out.name, "seconds": round(sec, 1), "utc": utc()}), flush=True)


def main_exp6(F, lock, th, splits):
    fams = family_list(lock)
    for k in splits:
        inp = bank_inputs(F, k)
        oodset = set(inp["ood"])
        for cond in ("random", "id_first9000", "id_burst10", "ood1pct", "ood5pct", "burst10_ood5pct"):
            f_out = OUT / f"exp6_s{k}_{cond}.parquet"
            if f_out.exists():
                continue
            order_ids = orders(inp, cond, 123)
            flag = np.array([s in oodset for s in order_ids])
            bidx = np.arange(len(order_ids)) // 256
            out, admits, sec = run_order(F, inp, order_ids, bidx, fams)
            pos = {s: i for i, s in enumerate(inp["ids_all"])}
            mcm = np.log(inp["mcm_all"][[pos[s] for s in order_ids]])
            out_m = {f"{n}+MCM": x + mcm for n, x in out.items() if n in fams}
            rows = operating_rows({**out, **out_m}, flag, order_ids, th, {"exp": "exp6", "split": k, "cond": cond, "seconds": sec})
            # erroneous admissions (final memory stage M, both views): concentration and after-admission effect
            idc = np.array([s.split("/")[2] if not f else "" for s, f in zip(order_ids, flag)])
            for v in ("B14", "L14"):
                adm = admits[v][:, 2]
                id_adm = adm & ~flag
                per = pd.Series(idc[id_adm]).value_counts()
                rows.append({"family": f"admission_{v}", "exp": "exp6", "split": k, "cond": cond,
                             "id_admission_rate": float(100 * id_adm.sum() / max((~flag).sum(), 1)),
                             "ood_admission_rate": float(100 * (adm & flag).sum() / max(flag.sum(), 1)),
                             "memory_id_fraction": float(100 * id_adm.sum() / max(adm.sum(), 1)),
                             "classes_with_id_admission": int(len(per)), "max_class_share": float(per.max() / max(per.sum(), 1)) if len(per) else 0.0})
            tmp = OUT / f"exp6_s{k}_{cond}.{os.getpid()}.parquet"
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, f_out)
            np.savez_compressed(OUT / f"exp6_s{k}_{cond}_scores.npz", ids=np.array(order_ids), flag=flag,
                                **{n.replace("+", "_"): x for n, x in out.items()}, admit_B14=admits["B14"], admit_L14=admits["L14"])
            print(json.dumps({"exp6": f_out.name, "seconds": round(sec, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["thresholds", "exp5", "exp6"], required=True)
    ap.add_argument("--seeds", default="123,124,125")
    ap.add_argument("--splits", default="1,2,3,4,5")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = load_lock()
    if a.exp == "thresholds":
        th = dev1_thresholds(lock)
        (OUT / "thresholds.json").write_text(json.dumps(th, indent=1))
        print(json.dumps(th))
    else:
        th = json.loads((OUT / "thresholds.json").read_text())
        F = Features()
        if a.exp == "exp5":
            main_exp5(F, lock, th, [int(s) for s in a.seeds.split(",")])
        else:
            main_exp6(F, lock, th, [int(s) for s in a.splits.split(",")])
