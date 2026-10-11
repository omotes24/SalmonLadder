"""Point 5 check: per-draw pooled-sigma FPR95 on dev1/dev2 under the Phase 14 protocol (3 orders = 3 fold assignments per draw)
and the Phase 16 protocol (ids sorted, one fold assignment per draw). Prints JSON."""
import glob, json, re
import numpy as np, pandas as pd
out = {}
K = "xfit|storey|q0.2|pp|pooled"
KP = "xfit|storey|q0.2|pp"
for dev in ("dev1", "dev2"):
    for stream in ("near", "far"):
        p14 = {}
        for f in glob.glob(f"/home/omote/reprise_p14_20261011/results/sel/{dev}/{dev}_draw*_{stream}_seed*.csv"):
            m = re.search(r"draw(\d+)_\w+_seed(\d+)", f)
            d = pd.read_csv(f)
            r = d[(d.cfg == K) & (d.base == "none")].iloc[0]
            rp = d[(d.cfg == KP) & (d.base == "none")].iloc[0]
            p14.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(r.FPR95), float(rp.FPR95)))
        p16 = {}
        for f in glob.glob(f"/home/omote/reprise_p16_20261011/results/h16_dev/{dev}/{dev}_draw*_{stream}.csv"):
            m = re.search(r"draw(\d+)_", f)
            d = pd.read_csv(f)
            r = d[(d.cfg == K) & (d.base == "none")].iloc[0]
            rp = d[(d.cfg == KP) & (d.base == "none")].iloc[0]
            p16[int(m.group(1))] = (float(r.FPR95), float(rp.FPR95))
        rows = []
        for dr in sorted(p14):
            v = sorted(p14[dr])
            rows.append({"draw": dr, "p14_sigma_per_order": [round(x[1], 3) for x in v], "p14_pfin_per_order": [round(x[2], 3) for x in v],
                         "p16_sigma": round(p16.get(dr, (np.nan, np.nan))[0], 3), "p16_pfin": round(p16.get(dr, (np.nan, np.nan))[1], 3)})
        a14 = np.mean([np.mean([x[1] for x in p14[dr]]) for dr in p14])
        a16 = np.mean([p16[dr][0] for dr in p16])
        spread = np.mean([max(x[1] for x in p14[dr]) - min(x[1] for x in p14[dr]) for dr in p14])
        out[f"{dev}|{stream}"] = {"rows": rows, "mean_p14_sigma": round(a14, 3), "mean_p16_sigma": round(a16, 3), "mean_range_over_orders": round(spread, 3)}
print(json.dumps(out, indent=1))
