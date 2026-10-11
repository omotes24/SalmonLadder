"""Phase 16 exploration (dev1 only): what the degree factor of the Phase 14 mass carries. Per task and view: Spearman(u+, sqrt deg)
over the test nodes; FPR95 of the degree alone, of u+, and of u+/sqrt(deg) (the walk-probability part). Output: results/dev1_e."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import p14common as P  # noqa: E402
import core14 as K  # noqa: E402
from an11 import measures  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402
from harm16 import sym_W  # noqa: E402
P16 = Path("/home/omote/reprise_p16_20261011")
out_dir = P16 / "results" / "dev1_e"; out_dir.mkdir(parents=True, exist_ok=True)
D = dev5.load("dev1")
seen, tasks = set(), []
for k in sorted(D["tins"].keys()):
    if (k[0], k[1]) not in seen:
        seen.add((k[0], k[1])); tasks.append(k)
rows = []
for (draw, stream, seed) in tasks:
    sid, flag, S_, _ = D["tins"][(draw, stream, seed)]; flag = np.asarray(flag, bool)
    r_ = np.array([D["stream_row"][s] for s in sid])
    for v in ("L14", "D3L"):
        sup, cal, sf = dev5.raw_view(D, draw, v, r_)
        sup, cal, sf = P.l2n(sup.reshape(-1, sup.shape[-1])), P.l2n(cal), P.l2n(sf)
        g = K.Graph(sup, cal, sf, "cuda")
        u = g.mass(np.arange(g.ns))
        W, deg = sym_W(g.graph); deg = deg.cpu().numpy()
        te = g.nfix + np.arange(g.nt)
        rho = spearmanr(u[te], np.sqrt(deg[te]))[0]
        r = {"draw": draw, "stream": stream, "view": v, "spearman_u_sqrtdeg": float(rho)}
        for name, x in (("deg", deg[te]), ("u", u[te]), ("u_over_sqrtdeg", u[te] / np.sqrt(deg[te]))):
            au, fp = measures(x, flag); r[f"{name}_AUROC"], r[f"{name}_FPR95"] = au, fp
        rows.append(r); g.free(); torch.cuda.empty_cache()
        print(json.dumps(r), flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv(out_dir / "degree_dev1.csv", index=False)
print("HARM16E_DONE", flush=True)
