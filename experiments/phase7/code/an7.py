"""Phase 7 analysis helpers (labels are used here only).

A result file holds the log p read-outs of ONE view on one stream; a view set is the product over its views.
Families (sum over the views of the set):
  static        static p                         mem      frozen memory p_M
  plp           frozen propagation (warm)        zeta     robust z of the zero-start u (best minimal configuration)
  static_plp    static p x p_LP                  reprise  p_M x p_LP (frozen REPRISE v5)
Candidates of experiment A: (memory read-out, static weight a, propagation read-out, truncation tau)."""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import p7common  # noqa: F401  (paths of the Phase 4 / 5 modules)
from metrics_p4 import metrics, tci

RESULTS = Path(os.environ.get("P7_RESULTS", "/home/omote/reprise_p7_20261003/results"))
BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
FROZEN = ("M", 0.0, "lp0", 1.0)


class Z:
    def __init__(self, path):
        self.z = np.load(path, allow_pickle=True)
        self.files = self.z.files
        self.c = {}

    def __getitem__(self, k):
        if k not in self.c:
            self.c[k] = self.z[k]
        return self.c[k]


def tasks(exp, view, pattern="*"):
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(str(RESULTS / exp / view / f"{pattern}.npz")))


class Stream:
    """All views of one stream; per-view read-outs via v(view, key), sums via s(key)."""

    def __init__(self, exp, task, views):
        self.views = list(views)
        self.z = {v: Z(RESULTS / exp / v / f"{task}.npz") for v in views}
        z0 = self.z[views[0]]
        for v in views[1:]:
            assert np.array_equal(self.z[v]["sample_id"], z0["sample_id"]), (exp, task, v)
        self.is_ood = z0["is_ood"].astype(bool)
        self.cls = z0["cls"]
        self.bidx = z0["bidx"]
        self.logM = z0["logM"]
        self.logS = z0["logS"] if "logS" in z0.files else None
        self.task = task

    def v(self, view, key):
        return self.z[view][f"s::{key}"]

    def s(self, key):
        return sum(self.v(v, key) for v in self.views)

    def family(self, name):
        if name == "static":
            return self.s("static")
        if name == "mem":
            return self.s("M")
        if name == "plp":
            return self.s("lp0")
        if name == "zeta":
            return self.s("z")
        if name == "static_plp":
            return self.s("static") + self.s("lp0")
        if name == "reprise":
            return self.s("M") + self.s("lp0")
        raise KeyError(name)

    def cand(self, spec):
        """spec = (memory read-out, a, lp read-out, tau)."""
        mem, a, lp, tau = spec
        x = 0.0
        for v in self.views:
            st = self.v(v, "static")
            m = np.minimum(self.v(v, "M"), st) if mem == "Mmin" else self.v(v, mem)
            l = self.v(v, lp)
            if tau < 1.0:
                l = np.minimum(0.0, l - np.log(tau))
            x = x + m + l + (a * st if a else 0.0)
        return x

    def base(self, name):
        return {"none": 0.0, "TINS": self.logS, "MCM": self.logM}[name]

    def appearance(self):
        """k = arrival index within the unknown class (1-based) for OOD rows, 0 for ID rows."""
        if not hasattr(self, "_k"):
            k = np.zeros(len(self.is_ood), int)
            seen = {}
            for i in np.flatnonzero(self.is_ood):
                c = self.cls[i]
                seen[c] = seen.get(c, 0) + 1
                k[i] = seen[c]
            self._k = k
        return self._k


LOCK = Path("/home/omote/reprise_p7_20261003/selection_lock_p7.json")


def locked_spec():
    """The extension locked by amendment 02 (None before the lock)."""
    f = Path(os.environ.get("P7_LOCK", str(LOCK)))
    return tuple(json.loads(f.read_text())["spec"]) if f.exists() else None


def threshold(score, is_ood):
    id_s = np.sort(np.asarray(score, np.float64)[~is_ood])
    return id_s[len(id_s) - int(np.ceil(0.95 * len(id_s)))]


def by_bin(score, is_ood, k, bins=BINS):
    """Miss rate (score >= the ID-95% threshold) and AUROC against all ID images, per appearance bin."""
    thr = threshold(score, is_ood)
    r = rankdata(score)
    out = {}
    n_id = int((~is_ood).sum())
    for lo, hi in bins:
        sel = is_ood & (k >= lo) & (k <= hi)
        if not sel.any():
            out[f"{lo}-{hi}"] = {"miss": np.nan, "auroc": np.nan, "n": 0}
            continue
        y = np.r_[np.ones(n_id, bool), np.zeros(int(sel.sum()), bool)]
        out[f"{lo}-{hi}"] = {"miss": 100 * float(np.mean(score[sel] >= thr)),
                             "auroc": 100 * float(roc_auc_score(y, np.r_[r[~is_ood], r[sel]])), "n": int(sel.sum())}
    return out


def unit_mean(rows, unit, key):
    """rows: list of dicts with the unit id; returns the per-unit means of rows[key] in unit order."""
    us = sorted({r[unit] for r in rows})
    return np.array([np.mean([r[key] for r in rows if r[unit] == u]) for u in us]), us


def paired(rows_a, rows_b, unit, key):
    a, ua = unit_mean(rows_a, unit, key)
    b, ub = unit_mean(rows_b, unit, key)
    assert ua == ub
    return tci(a - b)


def fmt_ci(d, nd=2):
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def parse_std(task):
    """U1_s3_d1_seed124 -> (split 3, draw 1, seed 124); U2_s0_d4_seed123 likewise."""
    m = re.match(r"(\w+?)_s(\d+)_d(\d+)_seed(\d+)$", task)
    return {"bank": m.group(1), "split": int(m.group(2)), "draw": int(m.group(3)), "seed": int(m.group(4))}


def parse_dev(task):
    m = re.match(r"(dev\d)_draw(\d+)_(\w+)_seed(\d+)$", task)
    return {"dev": m.group(1), "draw": int(m.group(2)), "stream": m.group(3), "seed": int(m.group(4))}


def dumpj(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")
