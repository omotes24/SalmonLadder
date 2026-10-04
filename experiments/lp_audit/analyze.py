"""Read-only diagnostics and paired summaries; never changes detector settings."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from metrics import ci,metrics
from report import paired
H=Path(__file__).resolve().parent;O=H/'reports'


def identity(name):
    d,s,r=name.replace('draw','').replace('seed','').split('_')
    return dict(draw=int(d),stream=s,seed=int(r))


def complete(g):
    return len(g)==15 and g.draw.nunique()==5 and all(g.groupby('draw').seed.nunique()==3)


def batch_auc(score,flag):
    m=int((~flag).sum());n=int(flag.sum());r=rankdata(score)
    return 100*(r[~flag].sum()-m*(m+1)/2)/(m*n)


def primary_diagnostics():
    numeric=[];view_metrics=[];admissions=[];finite=[];rank_bounds=[]
    for path in (H/'results/primary').glob('draw*.npz'):
        if 'calibration' in path.name:continue
        if not path.with_suffix('.json').exists():continue
        ident=identity(path.stem);z=np.load(path,allow_pickle=True);flag=z['is_ood'];bi=z['batch_index']
        cal=np.load(path.with_name(path.stem+'_calibration.npz'))
        solver=pd.read_csv(path.with_name(path.stem+'_solver.csv'))
        for key in z.files:
            if not key.startswith('log_score_'):continue
            a=z[key];finite.append(dict(**ident,score=key,NaN=int(np.isnan(a).sum()),positive_infinity=int(np.isposinf(a).sum()),
                                        log_zero=int(np.isneginf(a).sum())))
        for view in ['B14','L14']:
            for mode in ['L0','L1','L2']:
                raw=z[f'{view}_{mode}_raw'];p=z[f'{view}_{mode}_plp']
                trace=solver[(solver.view==view)&(solver.solver==mode)].set_index('batch')
                calibration_matrix=cal[f'{view}_{mode}']
                for index,batch in enumerate(np.unique(bi)):
                    ix=bi==batch;delta=float(trace.loc[batch,'l2_error_bound']);cu=np.sort(calibration_matrix[index])
                    ambiguities=np.searchsorted(cu,raw[ix]+2*delta,side='right')-np.searchsorted(cu,raw[ix]-2*delta,side='left')
                    rank_bounds.append(dict(**ident,view=view,solver=mode,batch=int(batch),coordinate_bound=delta,
                        zero_ambiguous_fraction=float(np.mean(ambiguities==0)),mean_p_error_bound=float(np.mean(ambiguities)/(len(cu)+1)),
                        max_p_error_bound=float(np.max(ambiguities)/(len(cu)+1)),
                        status='conditional on exact symmetric-normalization norm bound; not a machine certificate'))
                a=metrics(raw,flag);b=metrics(p,flag)
                within=[]
                for batch in np.unique(bi):
                    ix=bi==batch
                    if flag[ix].any() and (~flag[ix]).any():
                        within.append((batch_auc(raw[ix],flag[ix]),batch_auc(p[ix],flag[ix]),int(flag[ix].sum()*(~flag[ix]).sum())))
                w=np.array(within)
                view_metrics.append(dict(**ident,view=view,solver=mode,raw_AUROC=a['AUROC'],p_AUROC=b['AUROC'],
                    within_batch_raw_AUROC=float(np.average(w[:,0],weights=w[:,2])),within_batch_p_AUROC=float(np.average(w[:,1],weights=w[:,2])),
                    raw_FPR95=a['FPR95'],p_FPR95=b['FPR95']))
            for mode in ['L1','L2']:
                u=z[f'{view}_{mode}_raw'];u0=z[f'{view}_L0_raw'];p=z[f'{view}_{mode}_plp'];p0=z[f'{view}_L0_plp']
                numeric.append(dict(**ident,view=view,solver=mode,max_abs_u=float(np.max(np.abs(u-u0))),
                    mean_abs_u=float(np.mean(np.abs(u-u0))),max_abs_p=float(np.max(np.abs(p-p0))),
                    changed_p_fraction=float(np.mean(p!=p0)),spearman_u=float(spearmanr(u,u0).statistic)))
            for memory in ['M0','M1']:
                mask=z[f'{view}_{memory}_admit2'];nmem=int(mask.sum())
                admissions.append(dict(**ident,view=view,memory=memory,n_admitted=nmem,n_ID_admitted=int(np.sum(mask&~flag)),
                    n_OOD_admitted=int(np.sum(mask&flag)),ID_admission_rate=100*float(mask[~flag].mean()),
                    OOD_admission_rate=100*float(mask[flag].mean()),final_memory_ID_fraction=100*float((mask&~flag).sum()/max(nmem,1))))
        # Per-element check also works if legitimate zeros occur elsewhere in the array.
        for key in z.files:
            if not key.startswith('B14_') or key.replace('B14_','L14_',1) not in z:continue
            a=z[key];b=z[key.replace('B14_','L14_',1)]
            if a.dtype==bool:continue
            underflow=(a>0)&(b>0)&(a*b==0)
            if underflow.any():raise FloatingPointError(f'{path.name}: {key}: {underflow.sum()} underflows')
    for name,data in [('solver_score_changes',numeric),('single_view_rank_metrics',view_metrics),('memory_admissions',admissions)]:
        pd.DataFrame(data).to_csv(O/f'{name}.csv',index=False)
    rank_pairs=[]
    for key,g in pd.DataFrame(view_metrics).groupby(['stream','view','solver']):
        for scope,a,b in [('global','p_AUROC','raw_AUROC'),('within_batch','within_batch_p_AUROC','within_batch_raw_AUROC')]:
            delta=(g[a]-g[b]).groupby(g['draw']).mean()
            rank_pairs.append(dict(zip(['stream','view','solver'],key),scope=scope,metric='AUROC pLP-minus-raw',**ci(delta)))
    pd.DataFrame(rank_pairs).to_csv(O/'single_view_paired_differences.csv',index=False)
    pd.DataFrame(finite).to_csv(O/'numeric_score_audit.csv',index=False)
    pd.DataFrame(rank_bounds).to_csv(O/'residual_rank_bounds.csv',index=False)
    if admissions:
        d=pd.DataFrame(admissions);out=[]
        for key,g in d.groupby(['stream','view','memory']):
            for m in ['ID_admission_rate','OOD_admission_rate','final_memory_ID_fraction']:
                v=g.groupby('draw')[m].mean();out.append(dict(zip(['stream','view','memory'],key),metric=m,**ci(v)))
        pd.DataFrame(out).to_csv(O/'memory_admission_summary.csv',index=False)


def solver_summary():
    p=O/'solver_audit.csv'
    if not p.exists():return
    d=pd.read_csv(p);ids=d.run.apply(identity);d['draw']=ids.apply(lambda x:x['draw']);d['stream']=ids.apply(lambda x:x['stream']);d['seed']=ids.apply(lambda x:x['seed'])
    # Both views run serially in each method. Shared graph construction is counted once per view.
    out=[];runs=[]
    for key,g in d.groupby(['draw','stream','seed','solver']):
        batch=g.groupby('batch')[['solve_seconds','graph_seconds']].sum()
        z=dict(zip(['draw','stream','seed','solver'],key));z.update(
            solve_seconds=float(g.solve_seconds.sum()),graph_seconds=float(g.graph_seconds.sum()),
            solve_mean_batch_ms=1000*batch.solve_seconds.mean(),solve_p95_batch_ms=1000*batch.solve_seconds.quantile(.95),
            cache_mean_batch_ms=1000*(batch.solve_seconds+batch.graph_seconds).mean(),
            relative_residual_max=g.relative_residual.max(),iterations_mean=g.iterations.mean(),
            rank_reversals=int(g.rank_reversals.sum()),raw_tie_fraction=g.raw_tie_fraction.mean(),p_tie_fraction=g.p_tie_fraction.mean(),
            unconverged=int((~g.converged).sum()),evaluations=len(g));runs.append(z)
    r=pd.DataFrame(runs);r.to_csv(O/'solver_run_summary.csv',index=False)
    for key,g in r.groupby(['stream','solver']):
        for m in ['solve_seconds','graph_seconds','solve_mean_batch_ms','solve_p95_batch_ms','cache_mean_batch_ms','relative_residual_max','iterations_mean','raw_tie_fraction','p_tie_fraction']:
            v=g.groupby('draw')[m].mean();out.append(dict(zip(['stream','solver'],key),metric=m,**ci(v)))
    pd.DataFrame(out).to_csv(O/'solver_summary.csv',index=False)
    dif=[]
    for stream in ['near','far']:
        sub=r[r.stream==stream].rename(columns={'solver':'method'})
        for method in ['L1','L2','class_L2']:
            for m in ['solve_seconds','solve_mean_batch_ms','cache_mean_batch_ms']:
                val=paired(sub,method,'L0',stream,m)
                if val:dif.append(dict(stream=stream,a=method,b='L0',metric=m,**val))
    pd.DataFrame(dif).to_csv(O/'solver_paired_time.csv',index=False)


def competing_methods():
    d=pd.read_csv(O/'all_run_metrics.csv');rows=[]
    for stage in ['primary','tuning','capacity','baselines']:
        part=d[d.stage==stage].copy()
        part['capacity']='all' if stage in ['primary','tuning'] else '8192'
        if stage=='baselines':part['capacity']=part.configuration.map({'all':'all','cap8192':'8192'})
        for _,r in part.iterrows():
            method=r['method'];cfg=r['configuration']
            if stage=='baselines':
                base,tail=method.split('_cfg');idx,suffix=(tail.split('+',1)+[''])[:2] if '+' in tail else (tail,'')
                method=base+('+'+suffix if suffix else '');cfg='cfg'+idx
            elif stage=='capacity':cfg='cfg0'
            rows.append({**r.to_dict(),'method':method,'configuration':cfg})
    table=pd.DataFrame(rows);table['candidate']=table.method+'@'+table.configuration
    table.to_csv(O/'matched_resource_comparison.csv',index=False)
    selected=[];means=[]
    for (capacity,method),g in table[table.stream=='near'].groupby(['capacity','method']):
        good={config:part.FPR95.mean() for config,part in g.groupby('configuration') if complete(part)}
        if not good:continue
        best=min(good,key=lambda x:(good[x],x));chosen=table[(table.capacity==capacity)&(table.method==method)&(table.configuration==best)]
        selected.append(dict(capacity=capacity,method=method,configuration=best,near_FPR95=good[best],completed_configs=len(good),
                             planned_configs=1 if capacity=='8192' and not method.endswith('_type') and '_type+' not in method else 3))
        for stream,s in chosen.groupby('stream'):
            if not complete(s):continue
            for metric in ['FPR95','AUROC']:
                means.append(dict(capacity=capacity,method=method,configuration=best,stream=stream,metric=metric,**ci(s.groupby('draw')[metric].mean())))
    pd.DataFrame(selected).to_csv(O/'selected_tuned_methods.csv',index=False)
    pd.DataFrame(means).to_csv(O/'tuned_means.csv',index=False)
    selection=[];deltas=[];any_selection=[];any_deltas=[]
    for capacity in ['all','8192']:
        for suffix in ['', '+TINS']:
            g=table[(table.capacity==capacity)&(table.stream=='near')]
            simple=g[g.method.str.endswith(('raw'+suffix,'norm'+suffix,'cdf'+suffix))]
            eligible={key:part.FPR95.mean() for key,part in simple.groupby('candidate') if complete(part)}
            if not eligible:continue
            best=min(eligible,key=lambda x:(eligible[x],x));selection.append(dict(capacity=capacity,fusion=suffix or 'standalone',candidate=best,near_FPR95=eligible[best],n_eligible=len(eligible)))
            sub=table[table.capacity==capacity].copy();sub['method']=sub.candidate
            targets=sub[(sub['method'].str.startswith('REPRISE_'))&sub['method'].str.contains(suffix+'@',regex=False)].method.unique()
            if not suffix:targets=[a for a in targets if '+TINS' not in a]
            for a in targets:
                for stream in ['near','far']:
                    for metric in ['FPR95','AUROC']:
                        val=paired(sub,a,best,stream,metric)
                        if val and val['n']==5:deltas.append(dict(capacity=capacity,a=a,b=best,stream=stream,metric=metric,**val))
            # Diagnostic stronger envelope also includes current-calibration pLP.
            # Keep it separate from the preregistered simple-transform pool.
            lp=g[g.method.str.endswith(('raw'+suffix,'norm'+suffix,'cdf'+suffix,'plp'+suffix))]
            all_eligible={key:part.FPR95.mean() for key,part in lp.groupby('candidate') if complete(part)}
            any_best=min(all_eligible,key=lambda x:(all_eligible[x],x))
            any_selection.append(dict(capacity=capacity,fusion=suffix or 'standalone',candidate=any_best,near_FPR95=all_eligible[any_best],
                                      scope='all LP without memory, including pLP; exploratory envelope'))
            for a in targets:
                for stream in ['near','far']:
                    for metric in ['FPR95','AUROC']:
                        val=paired(sub,a,any_best,stream,metric)
                        if val and val['n']==5:any_deltas.append(dict(capacity=capacity,a=a,b=any_best,stream=stream,metric=metric,**val))
    (O/'selected_competitive_lp.json').write_text(json.dumps(selection,indent=2))
    pd.DataFrame(deltas).to_csv(O/'candidate_vs_selected_lp.csv',index=False)
    (O/'strongest_lp_no_memory.json').write_text(json.dumps(any_selection,indent=2))
    pd.DataFrame(any_deltas).to_csv(O/'candidate_vs_no_memory_lp.csv',index=False)
    # All-history versus bounded comparison at the same original configuration.
    cap=table[(table.configuration=='cfg0')&table.stage.isin(['primary','capacity'])].copy()
    cap['method']=cap.method+'@'+cap.capacity
    out=[]
    for method in table[table.stage=='capacity'].method.unique():
        for stream in ['near','far']:
            for metric in ['FPR95','AUROC']:
                val=paired(cap,method+'@8192',method+'@all',stream,metric)
                if val:out.append(dict(method=method,stream=stream,metric=metric,**val))
    pd.DataFrame(out).to_csv(O/'capacity_paired_differences.csv',index=False)


def calibration_pairs():
    rows=[]
    for stage in ['synthetic','real_calibration']:
        ds=[]
        for p in (H/'results'/stage).glob('rep*.csv'):
            if '_solver' in p.name:continue
            d=pd.read_csv(p);d=d[d.window.isin(['all','ID','OOD','early','late'])]
            if 'view' not in d:d['view']='synthetic_one_view'
            ds.append(d)
        if not ds:continue
        df=pd.concat(ds,ignore_index=True)
        for key,g in df.groupby(['view','condition','alpha','window']):
            w=g.pivot(index='rep',columns='component',values='rate')
            for a,b in [('L1_plp','L0_plp'),('L2_plp','L0_plp'),('M1_pt','M0_pt'),('M1_admit2','M0_admit2'),
                        ('M0_pt','static'),('M1_pt','static')]:
                if a not in w or b not in w:continue
                v=(w[a]-w[b]).dropna()*100
                if len(v):rows.append(dict(stage=stage,**dict(zip(['view','condition','alpha','window'],key)),a=a,b=b,
                                          **{**ci(v),'unit':'paired replicate; percentage-point difference'}))
    pd.DataFrame(rows).to_csv(O/'calibration_paired_differences.csv',index=False)
    p=O/'real_detection_metrics.csv'
    if p.exists() and p.stat().st_size>1:
        d=pd.read_csv(p);means=[];diff=[]
        for (condition,method),g in d.groupby(['condition','method']):
            for metric in ['AUROC','FPR95']:means.append(dict(condition=condition,method=method,metric=metric,**{**ci(g[metric]),'unit':'calibration resample on fixed cohort'}))
        for condition,g in d.groupby('condition'):
            for metric in ['AUROC','FPR95']:
                w=g.pivot(index='rep',columns='method',values=metric)
                for a,b in [('L0_M0','L0_LP'),('L1_M0','L1_LP'),('L2_M0','L2_LP'),('L1_M1','L1_M0'),('L2_M1','L2_M0')]:
                    v=(w[a]-w[b]).dropna();diff.append(dict(condition=condition,a=a,b=b,metric=metric,**{**ci(v),'unit':'paired calibration resample'}))
        pd.DataFrame(means).to_csv(O/'real_detection_summary.csv',index=False);pd.DataFrame(diff).to_csv(O/'real_detection_paired.csv',index=False)


if __name__=='__main__':
    primary_diagnostics();solver_summary();competing_methods();calibration_pairs()
