"""Whole-stream numerical comparison to frozen v5 cached component predictions."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from metrics import metrics,log_nonnegative
H=Path(__file__).resolve().parent;B=Path('/home/omote/vins_gonogo_20260925/r5/dev1/round2/cache')


def main():
    rows=[]
    for p in (H/'results/primary').glob('draw*.npz'):
        if 'calibration' in p.name:continue
        if not p.with_suffix('.json').exists():continue
        draw,stream,seed=p.stem.replace('draw','').replace('seed','').split('_');z=np.load(p,allow_pickle=True)
        mem=B/f'mem_d{draw}_{stream}_{seed}_K20_m1_n048.npz'
        lp=B/f'lp_d{draw}_{stream}_{seed}_gamma1_kg10_lam0.9.npz'
        if not mem.exists() or not lp.exists():
            rows.append({'run':p.stem,'state':'MISSING_FROZEN_REFERENCE','memory_path':str(mem),'lp_path':str(lp)});continue
        a=np.load(mem);b=np.load(lp)
        ref=log_nonnegative(z['S'])
        for view in ['B14','L14']:
            for name,old in [('M0_pt',a[view]),('L0_plp',b[view])]:
                val=z[f'{view}_{name}'];delta=np.abs(val-old)
                rows.append(dict(run=p.stem,component=f'{view}_{name}',state='COMPARED',max_abs=float(delta.max()),
                                 changed_count=int(np.sum(delta>0)),n=len(delta)))
            ref=ref+log_nonnegative(a[view])+log_nonnegative(b[view])
        new=z['log_score_REPRISE_L0_M0+TINS'];mm=metrics(new,z['is_ood']);rr=metrics(ref,z['is_ood'])
        rows.append(dict(run=p.stem,component='REPRISE+TINS',state='COMPARED',delta_AUROC=mm['AUROC']-rr['AUROC'],delta_FPR95=mm['FPR95']-rr['FPR95']))
    pd.DataFrame(rows).to_csv(H/'reports/frozen_v5_equivalence.csv',index=False)


if __name__=='__main__':main()
