"""Post-primary diagnostic only: separate a static multiplicative factor from adaptive pt.

No model, graph, data, threshold, or candidate selection is changed. This control
was specified after primary results and is not a preregistered confirmatory test.
"""
from pathlib import Path
import numpy as np,pandas as pd
from metrics import metrics,log_nonnegative,ci
from report import paired
H=Path(__file__).resolve().parent;O=H/'reports'


def main():
    rows=[]
    for path in sorted((H/'results/primary').glob('draw*.npz')):
        if 'calibration' in path.name or not path.with_suffix('.json').exists():continue
        draw,stream,seed=path.stem.replace('draw','').replace('seed','').split('_')
        with np.load(path,allow_pickle=True) as z:
            flag=z['is_ood'];static=sum(log_nonnegative(z[f'{v}_static']) for v in ['B14','L14'])
            for mode in ['L0','L1','L2']:
                lp=sum(log_nonnegative(z[f'{v}_{mode}_plp']) for v in ['B14','L14'])
                for suffix in ['', '+TINS']:
                    tins=log_nonnegative(z['S']) if suffix else 0.
                    scores={f'{mode}_static_x_plp'+suffix:static+lp+tins,
                            f'REPRISE_{mode}_M0'+suffix:z[f'log_score_REPRISE_{mode}_M0'+suffix],
                            f'{mode}_plp'+suffix:lp+tins}
                    for method,score in scores.items():
                        rows.append(dict(draw=int(draw),seed=int(seed),stream=stream,method=method,**metrics(score,flag)))
    d=pd.DataFrame(rows);d.to_csv(O/'static_factor_metrics.csv',index=False)
    means=[];dif=[]
    for (stream,method),g in d.groupby(['stream','method']):
        for metric in ['AUROC','FPR95']:means.append(dict(stream=stream,method=method,metric=metric,**ci(g.groupby('draw')[metric].mean())))
    for mode in ['L0','L1','L2']:
        for suffix in ['', '+TINS']:
            for a,b in [(f'REPRISE_{mode}_M0'+suffix,f'{mode}_static_x_plp'+suffix),
                        (f'{mode}_static_x_plp'+suffix,f'{mode}_plp'+suffix)]:
                for stream in ['near','far']:
                    for metric in ['AUROC','FPR95']:dif.append(dict(a=a,b=b,stream=stream,metric=metric,**paired(d,a,b,stream,metric)))
    pd.DataFrame(means).to_csv(O/'static_factor_means.csv',index=False)
    pd.DataFrame(dif).to_csv(O/'static_factor_paired.csv',index=False)
    print('Static-factor diagnostic completed from the existing 30 primary score files.',flush=True)


if __name__=='__main__':main()
