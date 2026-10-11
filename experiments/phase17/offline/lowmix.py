"""Point 4 (d)(e): seeds' ID fraction against the mixing rate, and the cells with r <= 5, from the Phase 14 recurrence grid
(NINCO c in {5,10,20,64}, SSB-hard c in {...}, r in {1,2,5,10,20,30(50)}; 3 reps; Storey q = 0.2; both views)."""
import json
import numpy as np
T = json.load(open('/home/claude/sl_p19/data/p14_summary.json'))
R = T['rec']
rows = []
for k, e in R.items():
    ds, c, r = k.split('|')
    for v in ('view_L14', 'view_D3L'):
        s = e[v]['storey_q0.2']
        rows.append({"ds": ds, "c": int(c[1:]), "r": int(r[1:]), "mix": e['mixing_rate'], "view": v[5:], "seeds": s['seeds'], "ood_seeds": s['ood_seeds'],
                     "id_frac": s['id_frac'] if s['n_reps_with_seeds'] else None, "n_with": s['n_reps_with_seeds'],
                     "sigma": e['cfg']['xfit|storey|q0.2|pp|pooled|none']['FPR95'], "static": e['cfg']['xfit|static|pooled|none']['FPR95'],
                     "lp": e['cfg']['xfit|lp|pooled|none']['FPR95']})
out = {}
for lo, hi, tag in ((0, 0.01, 'lt1'), (0.01, 0.1, '1to10'), (0.1, 1.01, 'ge10')):
    sel = [x for x in rows if lo <= x['mix'] < hi and x['id_frac'] is not None]
    none_sel = [x for x in rows if lo <= x['mix'] < hi and x['id_frac'] is None]
    fr = np.array([x['id_frac'] for x in sel])
    out[tag] = {"cells_views": len(sel), "no_seed_cells_views": len(none_sel), "id_frac_mean": float(fr.mean()) if len(fr) else None,
                "id_frac_min": float(fr.min()) if len(fr) else None, "id_frac_max": float(fr.max()) if len(fr) else None,
                "seeds_median": float(np.median([x['seeds'] for x in sel])) if sel else None,
                "ood_seeds_median": float(np.median([x['ood_seeds'] for x in sel])) if sel else None}
# per view D3L, list the low-mixing cells
out['lt1_cells_D3L'] = sorted([(x['ds'], x['c'], x['r'], round(100 * x['mix'], 3), round(x['seeds'], 1), round(x['ood_seeds'], 1), None if x['id_frac'] is None else round(100 * x['id_frac'], 1)) for x in rows if x['view'] == 'D3L' and x['mix'] < 0.01], key=lambda t: t[3])
# r <= 5: sigma vs static p
cells = {}
for x in rows:
    cells[(x['ds'], x['c'], x['r'])] = x
le5 = [x for x in cells.values() if x['r'] <= 5]
out['r_le5'] = {"n": len(le5), "sigma_worse_than_static": sum(x['sigma'] > x['static'] for x in le5),
                "mean_diff_sigma_minus_static": float(np.mean([x['sigma'] - x['static'] for x in le5])),
                "max_diff": float(max(x['sigma'] - x['static'] for x in le5)),
                "by_r": {r: {"n": len([x for x in le5 if x['r'] == r]), "worse": sum(x['sigma'] > x['static'] for x in le5 if x['r'] == r),
                             "mean_diff": float(np.mean([x['sigma'] - x['static'] for x in le5 if x['r'] == r]))} for r in (1, 2, 5)}}
ge10 = [x for x in cells.values() if x['r'] >= 10]
out['r_ge10'] = {"n": len(ge10), "sigma_better_than_static": sum(x['sigma'] < x['static'] for x in ge10),
                 "mean_diff_sigma_minus_static": float(np.mean([x['sigma'] - x['static'] for x in ge10]))}
print(json.dumps(out, indent=1, default=float))
