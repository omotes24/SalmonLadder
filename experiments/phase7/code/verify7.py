"""Independent re-computation of the Phase 7 headline numbers from the saved scores (no an7 / metrics_p4 code paths):
own AUROC (Mann-Whitney with average ranks), own FPR95, own appearance index, own unit aggregation.
Compares with the summary JSON files written by the analysis scripts and prints the largest discrepancy."""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

R = Path(os.environ.get("P7_RESULTS", "/home/omote/reprise_p7_20261003/results"))
P4 = Path("/home/omote/reprise_p4_20260928")
LOCK = Path(os.environ.get("P7_LOCK", "/home/omote/reprise_p7_20261003/selection_lock_p7.json"))
SPEC = json.loads(LOCK.read_text())["spec"] if LOCK.exists() else None      # [memory read-out, a, propagation read-out, tau]
worst = {}


def ext(zs):
    """The locked extension, written out directly: per view log p of the memory read-out + log p of the propagation read-out."""
    mem, a, lp, tau = SPEC
    assert a == 0.0 and tau == 1.0, SPEC
    return sum(z[f"s::{mem}"] + z[f"s::{lp}"] for z in zs)


def auroc(score, ood):
    r = rankdata(score)
    n_id, n_ood = int((~ood).sum()), int(ood.sum())
    return 100.0 * (r[~ood].sum() - n_id * (n_id + 1) / 2) / (n_id * n_ood)


def fpr95(score, ood):
    ids = np.sort(score[~ood])[::-1]                    # descending
    thr = ids[int(np.ceil(0.95 * len(ids))) - 1]        # the 95% most ID-like images are accepted
    return 100.0 * float(np.mean(score[ood] >= thr)), thr


def ci(x):
    x = np.asarray(x, float)
    h = student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def load(exp, task, views=("B14", "L14")):
    zs = [np.load(R / exp / v / f"{task}.npz", allow_pickle=True) for v in views]
    return zs


def note(tag, mine, theirs):
    d = abs(mine - theirs)
    worst[tag] = max(worst.get(tag, 0.0), d)
    return d


def appearance(ood, cls):
    k = np.zeros(len(ood), int)
    c = {}
    for i in np.flatnonzero(ood):
        c[cls[i]] = c.get(cls[i], 0) + 1
        k[i] = c[cls[i]]
    return k


def check_u1():
    f = R / "an_u1.json"
    if not f.exists():
        return
    J = json.loads(f.read_text())
    per = {}
    for p in sorted(glob.glob(str(R / "u1std" / "B14" / "*.npz"))):
        t = os.path.basename(p)[:-4]
        s = int(re.search(r"_s(\d+)_", t).group(1))
        zs = load("u1std", t)
        ood = zs[0]["is_ood"].astype(bool)
        rep = sum(z["s::M"] + z["s::lp0"] for z in zs)
        zeta = sum(z["s::z"] for z in zs)
        st = sum(z["s::static"] for z in zs)
        k = appearance(ood, zs[0]["cls"])
        f_rep, thr = fpr95(rep, ood)
        per.setdefault(s, []).append({"rep_f": f_rep, "rep_a": auroc(rep, ood), "zeta_f": fpr95(zeta, ood)[0], "st_f": fpr95(st, ood)[0],
                                      "rep_T": fpr95(rep + zs[0]["logS"], ood)[0], "k1": 100 * float(np.mean(rep[ood & (k == 1)] >= thr))})
    m = {key: np.array([np.mean([r[key] for r in per[s]]) for s in sorted(per)]) for key in ("rep_f", "rep_a", "zeta_f", "st_f", "rep_T", "k1")}
    print(f"U1 ({sum(len(v) for v in per.values())} streams): REPRISE {m['rep_a'].mean():.2f} / {m['rep_f'].mean():.2f}, zeta FPR95 {m['zeta_f'].mean():.2f}, "
          f"static {m['st_f'].mean():.2f}, x TINS {m['rep_T'].mean():.2f}, REPRISE - zeta {ci(m['rep_f'] - m['zeta_f'])}, k=1 miss {m['k1'].mean():.2f}")
    note("u1 reprise FPR95", m["rep_f"].mean(), J["overall"]["reprise"]["none"]["FPR95"]["mean"])
    note("u1 reprise AUROC", m["rep_a"].mean(), J["overall"]["reprise"]["none"]["AUROC"]["mean"])
    note("u1 zeta FPR95", m["zeta_f"].mean(), J["overall"]["zeta"]["none"]["FPR95"]["mean"])
    note("u1 reprise-zeta", ci(m["rep_f"] - m["zeta_f"])[0], J["diffs"]["reprise-zeta"]["FPR95"]["mean"])
    note("u1 reprise-zeta lo", ci(m["rep_f"] - m["zeta_f"])[1], J["diffs"]["reprise-zeta"]["FPR95"]["lo"])
    note("u1 k=1 miss", m["k1"].mean(), J["bins"]["reprise"]["1-1"]["miss"]["mean"])
    note("u1 reprise FPR95 vs Phase 4 (26.27)", m["rep_f"].mean(), 26.27)
    note("u1 zeta FPR95 vs Phase 4 (29.43)", m["zeta_f"].mean(), 29.43)


def check_dev():
    f = R / "a_dev1_selection.json"
    if not f.exists():
        return
    J = json.loads(f.read_text())
    for dev, ref in (("dev1", J), ("dev2", json.loads((R / "a_dev2_confirm_1.json").read_text()) if (R / "a_dev2_confirm_1.json").exists() else None)):
        if ref is None:
            continue
        per = {}
        for p in sorted(glob.glob(str(R / "dev" / "B14" / f"{dev}_draw*_near_seed*.npz"))):
            t = os.path.basename(p)[:-4]
            d = int(re.search(r"draw(\d+)", t).group(1))
            zs = load("dev", t)
            ood = zs[0]["is_ood"].astype(bool)
            k = appearance(ood, zs[0]["cls"])
            rep = sum(z["s::M"] + z["s::lp0"] for z in zs)
            cand = sum(z["s::Mh2"] + z["s::lp"] for z in zs)
            fr, tr = fpr95(rep, ood)
            fc, tc = fpr95(cand, ood)
            per.setdefault(d, []).append({"f": fr, "F1": 100 * float(np.mean(rep[ood & (k == 1)] >= tr)), "fc": fc,
                                          "F1c": 100 * float(np.mean(cand[ood & (k == 1)] >= tc))})
        m = {key: np.array([np.mean([r[key] for r in per[d]]) for d in sorted(per)]) for key in ("f", "F1", "fc", "F1c")}
        print(f"{dev}: frozen FPR95 {m['f'].mean():.2f} F1 {m['F1'].mean():.2f}; Mh2+lp FPR95 {m['fc'].mean():.2f} F1 {m['F1c'].mean():.2f}; dF1 {ci(m['F1c'] - m['F1'])}")
        fz = ref["frozen"]
        note(f"{dev} frozen FPR95", m["f"].mean(), fz["near_FPR95"])
        note(f"{dev} frozen F1", m["F1"].mean(), fz["F1"])
        if dev == "dev2":
            note("dev2 dF1", ci(m["F1c"] - m["F1"])[0], ref["difference"]["F1"]["mean"])
            note("dev2 dF1 hi", ci(m["F1c"] - m["F1"])[2], ref["difference"]["F1"]["hi"])


def check_retro():
    for exp, unit in (("u1r", "s"), ("u4r", "s")):
        f = R / f"an_retro_{exp}.json"
        if not f.exists():
            continue
        J = json.loads(f.read_text())
        per = {}
        for p in sorted(glob.glob(str(R / exp / "B14" / "*.npz"))):
            t = os.path.basename(p)[:-4]
            s = int(re.search(r"_s(\d+)_", t).group(1))
            zs = load(exp, t)
            ood = zs[0]["is_ood"].astype(bool)
            k = appearance(ood, zs[0]["cls"])
            row = {}
            for dl in ("", "@5", "@end"):
                x = sum(z[f"s::M{dl}"] + z[f"s::lp0{dl}"] for z in zs)
                fx, thr = fpr95(x, ood)
                row[f"f{dl}"], row[f"k1{dl}"] = fx, 100 * float(np.mean(x[ood & (k == 1)] >= thr))
            per.setdefault(s, []).append(row)
        m = {key: np.array([np.mean([r[key] for r in per[s]]) for s in sorted(per)]) for key in per[sorted(per)[0]][0]}
        print(f"{exp}: k=1 miss issued {m['k1'].mean():.2f}, delay 5 {m['k1@5'].mean():.2f}, end {m['k1@end'].mean():.2f}; FPR95 {m['f'].mean():.2f} -> end {m['f@end'].mean():.2f}; "
              f"d(k1, 5) {ci(m['k1@5'] - m['k1'])}")
        note(f"{exp} k1 issued", m["k1"].mean(), J["fam"]["reprise"]["0"]["miss"]["1-1"]["mean"])
        note(f"{exp} k1 delay5", m["k1@5"].mean(), J["fam"]["reprise"]["5"]["miss"]["1-1"]["mean"])
        note(f"{exp} k1 end", m["k1@end"].mean(), J["fam"]["reprise"]["end"]["miss"]["1-1"]["mean"])
        note(f"{exp} FPR95 end", m["f@end"].mean(), J["fam"]["reprise"]["end"]["FPR95"]["mean"])
        note(f"{exp} dF1 delay5 hi", ci(m["k1@5"] - m["k1"])[2], J["fam"]["reprise"]["5"]["dF1"]["hi"])


def check_c1():
    f = R / "c1_summary.json"
    if not f.exists():
        return
    from sklearn.metrics import roc_auc_score
    J = json.loads(f.read_text())
    src = P4 / "results" / "exp4"
    acc = {r: [] for r in (1, 20)}
    for k in range(1, 6):
        dz = json.loads((P4 / "banks" / "exp4" / f"split{k}.json").read_text())
        idq = dz["id_queries"]
        b = pd.read_parquet(src / f"s{k}_r0.parquet")
        b = {fam: g.set_index("query").score for fam, g in b.groupby("family")}
        for w in dz["probe"]:
            qs = dz["probes"][w]["queries"]
            p = pd.read_parquet(src / f"s{k}_{w}.parquet")
            p = p[p.cond == "same"]
            y = np.r_[np.ones(len(idq)), np.zeros(len(qs))]
            q = idq + qs
            a00 = 100 * roc_auc_score(y, (b["Mpt"].loc[q] + b["pLP"].loc[q]).values)
            for r in acc:
                g = {fam: h.set_index("query").score for fam, h in p[p.r == r].groupby("family")}
                am = 100 * roc_auc_score(y, (g["Mpt"].loc[q] + b["pLP"].loc[q]).values)
                ag = 100 * roc_auc_score(y, (b["Mpt"].loc[q] + g["pLP"].loc[q]).values)
                ab = 100 * roc_auc_score(y, (g["Mpt"].loc[q] + g["pLP"].loc[q]).values)
                acc[r].append((am - a00, ag - a00, ab - a00, ab - am - ag + a00))
    for rec in J["table"]:
        if rec["cond"] == "same" and rec["r"] in acc:
            mine = np.mean(acc[rec["r"]], axis=0)
            print(f"C1 r={rec['r']}: E_M {mine[0]:+.2f} E_G {mine[1]:+.2f} E_MG {mine[2]:+.2f} I {mine[3]:+.2f}")
            for j, e in enumerate(("E_M", "E_G", "E_MG", "I")):
                note(f"C1 r={rec['r']} {e}", mine[j], rec[e]["mean"])


def check_generic(json_name, exp, key_fn, pick):
    """Re-compute REPRISE / static (and the extension) AUROC and FPR95 for the cells chosen by `pick` in a summary JSON."""
    f = R / json_name
    if not f.exists():
        return
    J = json.loads(f.read_text())
    for cell_key, getter in pick(J):
        ref = getter()
        has_ext = SPEC is not None and "ext|none" in ref
        src = exp + "w" if has_ext else exp
        files = [p for p in sorted(glob.glob(str(R / src / "B14" / "*.npz"))) if key_fn(os.path.basename(p)[:-4], cell_key)]
        per = {}
        for p in files:
            t = os.path.basename(p)[:-4]
            if not (R / src / "L14" / f"{t}.npz").exists():
                continue
            zs = load(src, t)
            ood = zs[0]["is_ood"].astype(bool)
            rep = sum(z["s::M"] + z["s::lp0"] for z in zs)
            st = sum(z["s::static"] for z in zs)
            u = re.search(r"(?:^|_)s(\d+)_", t)
            u = int(u.group(1)) if u else int(re.search(r"seed(\d+)", t).group(1))
            row = [fpr95(rep, ood)[0], auroc(rep, ood), fpr95(st, ood)[0]]
            if has_ext:
                e = ext(zs)
                row += [fpr95(e, ood)[0], auroc(e, ood)]
            per.setdefault(u, []).append(row)
        if not per:
            continue
        m = np.array([np.mean(per[u], axis=0) for u in sorted(per)])
        print(f"{json_name} {cell_key} [{src}]: REPRISE {m[:, 1].mean():.2f} / {m[:, 0].mean():.2f} static FPR95 {m[:, 2].mean():.2f}"
              + (f" ext {m[:, 4].mean():.2f} / {m[:, 3].mean():.2f}" if has_ext else "") + f" (units {len(per)}, runs {sum(len(v) for v in per.values())})")
        note(f"{json_name} {cell_key} FPR95", m[:, 0].mean(), ref["reprise|none"]["FPR95"]["mean"])
        note(f"{json_name} {cell_key} AUROC", m[:, 1].mean(), ref["reprise|none"]["AUROC"]["mean"])
        note(f"{json_name} {cell_key} static", m[:, 2].mean(), ref["static|none"]["FPR95"]["mean"])
        if has_ext:
            note(f"{json_name} {cell_key} ext FPR95", m[:, 3].mean(), ref["ext|none"]["FPR95"]["mean"])
            note(f"{json_name} {cell_key} ext AUROC", m[:, 4].mean(), ref["ext|none"]["AUROC"]["mean"])
            d = ci(m[:, 3] - m[:, 0])
            note(f"{json_name} {cell_key} ext-REPRISE FPR95", d[0], ref["ext-reprise"]["FPR95"]["mean"])
            if len(per) > 1:
                note(f"{json_name} {cell_key} ext-REPRISE FPR95 hi", d[2], ref["ext-reprise"]["FPR95"]["hi"])


def pick_e(J):
    out = []
    for rec in J.get("e1", []):
        c = rec["cell"]
        if (c["n_id"], c["U"], c["m"]) in ((18000, 100, 1), (18000, 100, 50), (4500, 30, 20)):
            out.append((f"_U{c['U']}_m{c['m']}_n{c['n_id']}_random_B256", (lambda rec=rec: rec)))
    for rec in J.get("e2", []):
        c = rec["cell"]
        if c["pattern"] in ("ood_burst", "emerging"):
            m = {"native": 50, "sparse": 10}[c["comp"]]
            out.append((f"_U100_m{m}_n18000_{c['pattern']}_B256", (lambda rec=rec: rec)))
    return out


def pick_d(J):
    out = []
    for rec in J["rows"]:
        if rec["ds"] in ("cub", "places365", "insk") or rec["ds"] == "U1":
            out.append((f"{rec['ds']}|{rec['cond']}", (lambda rec=rec: rec)))
    return out


def check_lock():
    """Post-lock evaluation of the extension on U1 / U4 (amendment 02)."""
    f = R / "an_lock_eval.json"
    if not f.exists() or SPEC is None:
        return
    J = json.loads(f.read_text())
    assert J["spec"] == SPEC, (J["spec"], SPEC)
    for exp in ("u1w", "u4w"):
        per = {}
        for p in sorted(glob.glob(str(R / exp / "B14" / "*.npz"))):
            t = os.path.basename(p)[:-4]
            s_ = int(re.search(r"_s(\d+)_", t).group(1))
            zs = load(exp, t)
            ood = zs[0]["is_ood"].astype(bool)
            k = appearance(ood, zs[0]["cls"])
            rep, e = sum(z["s::M"] + z["s::lp0"] for z in zs), ext(zs)
            fr, tr = fpr95(rep, ood)
            fe, te = fpr95(e, ood)
            T = zs[0]["logS"]
            per.setdefault(s_, []).append({"fr": fr, "fe": fe, "ar": auroc(rep, ood), "ae": auroc(e, ood), "frT": fpr95(rep + T, ood)[0], "feT": fpr95(e + T, ood)[0],
                                           "k1r": 100 * float(np.mean(rep[ood & (k == 1)] >= tr)), "k1e": 100 * float(np.mean(e[ood & (k == 1)] >= te))})
        m = {key: np.array([np.mean([r[key] for r in per[u]]) for u in sorted(per)]) for key in per[sorted(per)[0]][0]}
        dA, dF, dK = ci(m["ae"] - m["ar"]), ci(m["fe"] - m["fr"]), ci(m["k1e"] - m["k1r"])
        print(f"{exp} lock evaluation ({sum(len(v) for v in per.values())} streams): AUROC {m['ar'].mean():.2f} -> {m['ae'].mean():.2f} ({dA[0]:+.2f} [{dA[1]:+.2f}, {dA[2]:+.2f}]); "
              f"FPR95 {m['fr'].mean():.2f} -> {m['fe'].mean():.2f} ({dF[0]:+.2f} [{dF[1]:+.2f}, {dF[2]:+.2f}]); k=1 miss {dK[0]:+.2f} [{dK[1]:+.2f}, {dK[2]:+.2f}]")
        Je = J[exp]
        note(f"{exp} lock frozen AUROC", m["ar"].mean(), Je["none|AUROC"]["frozen"]["mean"])
        note(f"{exp} lock ext AUROC", m["ae"].mean(), Je["none|AUROC"]["cand"]["mean"])
        note(f"{exp} lock dAUROC", dA[0], Je["none|AUROC"]["diff"]["mean"])
        note(f"{exp} lock dAUROC lo", dA[1], Je["none|AUROC"]["diff"]["lo"])
        note(f"{exp} lock ext FPR95", m["fe"].mean(), Je["none|FPR95"]["cand"]["mean"])
        note(f"{exp} lock dFPR95", dF[0], Je["none|FPR95"]["diff"]["mean"])
        note(f"{exp} lock dFPR95 hi", dF[2], Je["none|FPR95"]["diff"]["hi"])
        note(f"{exp} lock ext x TINS FPR95", m["feT"].mean(), Je["TINS|FPR95"]["cand"]["mean"])
        note(f"{exp} lock k=1 miss diff", dK[0], Je["bins"]["1-1"]["diff"]["mean"])


def check_within_dev():
    f = R / "a2_dev2_confirm.json"
    if not f.exists() or SPEC is None:
        return
    J = json.loads(f.read_text())
    per, far = {}, {}
    for kind, acc in (("near", per), ("far", far)):
        for p in sorted(glob.glob(str(R / "devw" / "B14" / f"dev2_draw*_{kind}_seed*.npz"))):
            t = os.path.basename(p)[:-4]
            d = int(re.search(r"draw(\d+)", t).group(1))
            zs = load("devw", t)
            ood = zs[0]["is_ood"].astype(bool)
            rep, e = sum(z["s::M"] + z["s::lp0"] for z in zs), ext(zs)
            acc.setdefault(d, []).append((auroc(e, ood) - auroc(rep, ood), fpr95(e, ood)[0] - fpr95(rep, ood)[0]))
    a = np.array([np.mean(per[d], axis=0) for d in sorted(per)])
    b = np.array([np.mean(far[d], axis=0) for d in sorted(far)])
    dA, dF = ci(a[:, 0]), ci(a[:, 1])
    print(f"dev2 confirmation of the extension: near AUROC {dA[0]:+.2f} [{dA[1]:+.2f}, {dA[2]:+.2f}], near FPR95 {dF[0]:+.2f} [{dF[1]:+.2f}, {dF[2]:+.2f}], far FPR95 {b[:, 1].mean():+.2f}")
    note("dev2 ext dAUROC", dA[0], J["difference"]["near_AUROC"]["mean"])
    note("dev2 ext dAUROC lo", dA[1], J["difference"]["near_AUROC"]["lo"])
    note("dev2 ext dFPR95", dF[0], J["difference"]["near_FPR95"]["mean"])
    note("dev2 ext dFPR95 hi", dF[2], J["difference"]["near_FPR95"]["hi"])
    note("dev2 ext far dFPR95", b[:, 1].mean(), J["difference"]["far_FPR95"]["mean"])


def check_cases():
    """Shares of the cases R / C / N1 / N+ with an own (table-based) implementation of their definition."""
    f = R / "an_cases_u1.json"
    if not f.exists():
        return
    J = json.loads(f.read_text())
    per = {}
    for p in sorted(glob.glob(str(R / "u1w" / "B14" / "*.npz"))):
        t = os.path.basename(p)[:-4]
        s_ = int(re.search(r"_s(\d+)_", t).group(1))
        zs = load("u1w", t)
        ood = zs[0]["is_ood"].astype(bool)
        df = pd.DataFrame({"cls": zs[0]["cls"], "b": zs[0]["bidx"], "ood": ood, "k": appearance(ood, zs[0]["cls"])})
        df = df[df.ood].copy()
        prev = np.zeros(len(df), bool)
        co = np.zeros(len(df), bool)
        for z in zs:
            df["adm"] = z["admit"][:, 2][ood].astype(bool)
            first = df[df.adm].groupby("cls").b.min()                       # batch of the first admitted image of each class
            prev |= (df.cls.map(first).fillna(np.inf) < df.b).values
            n_same = df.groupby(["cls", "b"]).adm.transform("sum") - df.adm    # admitted same-class images of the same batch, without the image itself
            co |= (n_same > 0).values
        case = np.where(prev, "R", np.where(co, "C", np.where(df.k.values == 1, "N1", "N+")))
        rep = sum(z["s::M"] + z["s::lp0"] for z in zs)
        thr = fpr95(rep, ood)[1]
        row = {c: 100 * float(np.mean(case == c)) for c in ("R", "C", "N1", "N+")}
        row["missC"] = 100 * float(np.mean(rep[ood][case == "C"] >= thr)) if (case == "C").any() else np.nan
        per.setdefault(s_, []).append(row)
    m = {key: np.array([np.nanmean([r[key] for r in per[u]]) for u in sorted(per)]) for key in ("R", "C", "N1", "N+", "missC")}
    print("U1 cases (own implementation): " + "  ".join(f"{c} {m[c].mean():.2f}%" for c in ("R", "C", "N1", "N+")) + f"; miss rate of case C (frozen) {m['missC'].mean():.2f}")
    for c in ("R", "C", "N1", "N+"):
        note(f"u1 case share {c}", m[c].mean(), J["share"][c]["mean"])
    note("u1 case C miss (frozen)", m["missC"].mean(), J["fam"]["reprise"]["case_miss"]["C"])


def check_b():
    f = R / "an_b.json"
    if not f.exists():
        return
    J = json.loads(f.read_text())
    for bank, unit_re in (("U1", r"_s(\d+)_"), ("U2", r"_d(\d+)_")):
        exp = J[bank]["exp"]
        for name in ("S14", "D3L", "G14", "SIG2L", "B14+L14", "L14+D3L", "S14+CLIP", "D3S", "D3SP", "RN50", "S14+D3S", "D3S+D3SP", "S14+RN50"):
            if name not in J[bank]["sets"]:
                continue
            views = name.split("+")
            per = {}
            for p in sorted(glob.glob(str(R / exp / views[0] / "*.npz"))):
                t = os.path.basename(p)[:-4]
                zs = load(exp, t, views)
                ood = zs[0]["is_ood"].astype(bool)
                rep, zt = sum(z["s::M"] + z["s::lp0"] for z in zs), sum(z["s::z"] for z in zs)
                row = [fpr95(rep, ood)[0], auroc(rep, ood), fpr95(zt, ood)[0]]
                if SPEC is not None and f"s::{SPEC[0]}" in zs[0].files:
                    e = ext(zs)
                    row += [fpr95(e, ood)[0], auroc(e, ood)]
                per.setdefault(int(re.search(unit_re, t).group(1)), []).append(row)
            m = np.array([np.mean(per[u], axis=0) for u in sorted(per)])
            ref = J[bank]["sets"][name]
            print(f"B {bank} {name:8s} [{exp}]: REPRISE {m[:, 1].mean():.2f} / {m[:, 0].mean():.2f}, REPRISE - zeta {ci(m[:, 0] - m[:, 2])[0]:+.2f}"
                  + (f", ext {m[:, 4].mean():.2f} / {m[:, 3].mean():.2f}" if m.shape[1] > 3 else ""))
            note(f"B {bank} {name} FPR95", m[:, 0].mean(), ref["reprise|none"]["FPR95"]["mean"])
            note(f"B {bank} {name} AUROC", m[:, 1].mean(), ref["reprise|none"]["AUROC"]["mean"])
            note(f"B {bank} {name} REPRISE-zeta", ci(m[:, 0] - m[:, 2])[0], ref["reprise-zeta"]["FPR95"]["mean"])
            note(f"B {bank} {name} REPRISE-zeta lo", ci(m[:, 0] - m[:, 2])[1], ref["reprise-zeta"]["FPR95"]["lo"])
            if m.shape[1] > 3 and "ext|none" in ref:
                note(f"B {bank} {name} ext FPR95", m[:, 3].mean(), ref["ext|none"]["FPR95"]["mean"])
                note(f"B {bank} {name} ext-REPRISE", ci(m[:, 3] - m[:, 0])[0], ref["ext-reprise"]["FPR95"]["mean"])


def cross_w():
    """The '<exp>w' runs repeat the frozen read-outs of the first runs: compare them file by file."""
    print("== frozen read-outs: first run vs the run with the within-batch read-out (same tasks)")
    ok = True
    for a_, b_, step in (("u1std", "u1w", 1), ("u4std", "u4w", 1), ("u2std", "u2w", 1), ("e1", "e1w", 4), ("e2", "e2w", 1), ("e3", "e3w", 1)):
        for v in ("B14", "L14"):
            fa = {os.path.basename(p) for p in glob.glob(str(R / a_ / v / "*.npz"))}
            fb = {os.path.basename(p) for p in glob.glob(str(R / b_ / v / "*.npz"))}
            common = sorted(fa & fb)[::step]
            if not common:
                continue
            stat = {k: [0, 0.0] for k in ("static", "M", "lp0", "lp", "z")}
            rows, dmax, same_stream = 0, 0.0, True
            for fn in common:
                za, zb = np.load(R / a_ / v / fn, allow_pickle=True), np.load(R / b_ / v / fn, allow_pickle=True)
                same_stream &= bool(np.array_equal(za["sample_id"], zb["sample_id"]) and np.array_equal(za["admit"], zb["admit"]))
                rows += len(za["is_ood"])
                for k in stat:
                    d = np.abs(za[f"s::{k}"] - zb[f"s::{k}"])
                    stat[k][0] += int((d > 1e-9).sum())
                    stat[k][1] = max(stat[k][1], float(d.max()))
                ood = za["is_ood"].astype(bool)
                dmax = max(dmax, abs(fpr95(za["s::M"] + za["s::lp0"], ood)[0] - fpr95(zb["s::M"] + zb["s::lp0"], ood)[0]))
            good = same_stream and stat["static"][0] == 0 and stat["M"][0] == 0 and dmax < 0.05
            ok &= good
            print(f"  {a_:6s} vs {b_:4s} {v}: {len(common)} files ({len(fa)} / {len(fb)}), {rows} rows; same stream and admissions {same_stream}; rows differing (max |diff|): "
                  + ", ".join(f"{k} {n} ({mx:.4f})" for k, (n, mx) in stat.items()) + f"; max |dFPR95| of one view's M x p_LP {dmax:.3f} -> {'ok' if good else 'DIFF'}")
    if not ok:
        worst["cross_w"] = 1.0


EXPECT = {"dev": 60, "devr": 30, "devw": 60, "u1std": 45, "u1r": 45, "u1w": 45, "u4std": 45, "u4r": 45, "u4w": 45, "u2std": 15, "u2w": 15, "u1alone": 5,
          "e1": 405, "e2": 70, "e3": 40, "e1w": 405, "e2w": 70, "e3w": 40, "dsw": 174}


def counts():
    print("== files per experiment / view (expected number in brackets)")
    for exp in sorted(os.listdir(R)):
        if (R / exp).is_dir():
            sub = {v: len(glob.glob(str(R / exp / v / "*.npz"))) for v in sorted(os.listdir(R / exp)) if (R / exp / v).is_dir()}
            if sub:
                bad = [v for v, n in sub.items() if exp in EXPECT and n != EXPECT[exp]]
                print(f"  {exp:8s} [{EXPECT.get(exp, '-')}] {sub}" + (f"   <-- incomplete: {bad}" if bad else ""))
                if bad:
                    worst[f"count {exp}"] = 1.0
            else:
                print(f"  {exp:8s} {len(os.listdir(R / exp))} files")


if __name__ == "__main__":
    counts()
    check_u1()
    check_dev()
    check_retro()
    check_c1()
    check_within_dev()
    check_lock()
    check_cases()
    check_b()
    check_generic("an_e.json", "e1", lambda t, k: k in t, lambda J: [x for x in pick_e(J) if "random" in x[0]])
    check_generic("an_e.json", "e2", lambda t, k: k in t, lambda J: [x for x in pick_e(J) if "random" not in x[0]])
    check_generic("an_d.json", "ds", lambda t, k: (t.startswith(k.split("|")[0] + "_s") and f"_{k.split('|')[1]}_" in t), pick_d)
    cross_w()
    print("== largest absolute discrepancies (own re-computation vs summary files)")
    for tag, d in sorted(worst.items(), key=lambda x: -x[1])[:80]:
        print(f"  {d:8.4f}  {tag}")
    bad = {k_: v for k_, v in worst.items() if v > 0.01 and "Phase 4" not in k_}
    print("VERIFY7_OK" if not bad else f"VERIFY7_DIFF {bad}")
