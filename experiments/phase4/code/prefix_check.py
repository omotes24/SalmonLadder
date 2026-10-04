"""Online validity check (Exp 5): issued scores never depend on later batches. Compares (a) the prefix run with the
full run (from exp5) and (b) two identical prefix runs, to separate future dependence from GPU floating-point
non-determinism (atomic sparse products can flip exact ties among calibration ranks)."""
import json

import numpy as np

from bank_data import Features
from common import RESULTS, utc
from evaluate import load_lock
from operating import bank_inputs, family_list, orders, run_order

if __name__ == "__main__":
    F = Features()
    lock = load_lock()
    fams = family_list(lock)
    inp = bank_inputs(F, 1)
    ids = orders(inp, "random", 123)
    npre = 36 * 256
    bidx = np.arange(len(ids)) // 256
    full, _, _ = run_order(F, inp, ids, bidx, fams)
    a, _, _ = run_order(F, inp, ids[:npre], bidx[:npre], fams)
    b, _, _ = run_order(F, inp, ids[:npre], bidx[:npre], fams)
    out = {"utc": utc(), "n_prefix": npre}
    for k in a:
        d_pf = np.abs(a[k] - full[k][:npre])
        d_pp = np.abs(a[k] - b[k])
        out[k] = {"prefix_vs_full_max": float(d_pf.max()), "prefix_vs_full_frac_nonzero": float((d_pf > 0).mean()),
                  "prefix_vs_prefix_max": float(d_pp.max()), "prefix_vs_prefix_frac_nonzero": float((d_pp > 0).mean())}
    (RESULTS / "prefix_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
