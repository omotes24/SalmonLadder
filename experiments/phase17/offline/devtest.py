"""Point 3: how well do the development splits predict the benchmark? The 18 Phase 14 candidates (seed rule x q x mode), each
evaluated on dev1, dev2 (5 draws) and the OpenOOD v1.5 ImageNet-1K test (6 draws), standalone. Spearman / Kendall rank
correlations of FPR95, the dev winners vs the test winners, and the pairwise reversals of the registered choices."""
import json, itertools
import numpy as np
from scipy.stats import spearmanr, kendalltau
D = json.load(open('/home/claude/sl_p19/data/sel14_summary.json'))
T = json.load(open('/home/claude/sl_p19/data/p14_summary.json'))['test_train']['openood']
out = {}
for stat in ("pfin", "sigma"):
    suf = "|none" if stat == "pfin" else "|pooled|none"
    cands = sorted(k[: -len(suf)] for k in D['dev1'] if k.startswith('xfit|') and k.endswith(suf) and (stat == 'sigma' or '|pooled' not in k)
                   and k[: -len(suf)].count('|') == 3)
    res = {"n": len(cands)}
    for g in ("near", "far"):
        d1 = np.array([D['dev1'][c + suf][g]['FPR95'] for c in cands])
        d2 = np.array([D['dev2'][c + suf][g]['FPR95'] for c in cands])
        te = np.array([T[g][c + suf]['FPR95'] for c in cands])
        res[g] = {"spearman_dev1_test": spearmanr(d1, te)[0], "spearman_dev2_test": spearmanr(d2, te)[0], "spearman_dev1_dev2": spearmanr(d1, d2)[0],
                  "kendall_dev2_test": kendalltau(d2, te)[0],
                  "best_dev1": cands[int(np.argmin(d1))], "best_dev2": cands[int(np.argmin(d2))], "best_test": cands[int(np.argmin(te))],
                  "rank_on_test_of_dev1_best": int(np.argsort(np.argsort(te))[int(np.argmin(d1))]) + 1}
        # the two registered choices of Phase 14 and their reversals on the test
        pairs = {"storey_vs_bh": ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp"), "q0.2_vs_q0.1": ("xfit|storey|q0.2|pp", "xfit|storey|q0.1|pp")}
        for name, (a, b) in pairs.items():
            ia, ib = cands.index(a), cands.index(b)
            res[g][name] = {"dev1": round(d1[ia] - d1[ib], 3), "dev2": round(d2[ia] - d2[ib], 3), "test": round(te[ia] - te[ib], 3)}
        # among all candidate pairs: share whose sign of the difference on dev2 agrees with the test
        agree, tot = 0, 0
        for i, j in itertools.combinations(range(len(cands)), 2):
            if abs(d2[i] - d2[j]) < 1e-9 or abs(te[i] - te[j]) < 1e-9:
                continue
            tot += 1
            agree += int(np.sign(d2[i] - d2[j]) == np.sign(te[i] - te[j]))
        res[g]["pair_sign_agreement_dev2_test"] = round(agree / tot, 3)
        # restricted to the 6 pp candidates (the mode chosen)
        pp = [i for i, c in enumerate(cands) if c.endswith('|pp')]
        res[g]["spearman_dev2_test_pp_only"] = spearmanr(d2[pp], te[pp])[0]
        res[g]["values"] = {c: [round(d1[i], 2), round(d2[i], 2), round(te[i], 2)] for i, c in enumerate(cands)}
    out[stat] = res
print(json.dumps(out, indent=1, default=float))
