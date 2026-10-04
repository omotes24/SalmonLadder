"""Aggregate paired development experiments. Selection never uses dev2/test."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
from metrics import ci,metrics,log_nonnegative
H=Path(__file__).resolve().parent;OUT=H/'reports';OUT.mkdir(exist_ok=True)


def paired(df,a,b,stream,column):
    sub=df[df.stream==stream]
    wide=sub.pivot(index=['draw','seed'],columns='method',values=column)
    if a not in wide or b not in wide:return None
    delta=(wide[a]-wide[b]).dropna();counts=delta.groupby(level='draw').count()
    delta=delta[delta.index.get_level_values('draw').isin(counts[counts==3].index)]
    if not len(delta):return None
    return ci(delta.groupby(level='draw').mean().values)


def aggregate_calibration(stage):
    paths=list((H/'results'/stage).glob('rep*.csv'))
    by=[]
    for p in paths:
        if '_solver' in p.name:continue
        d=pd.read_csv(p);d['stage']=stage;by.append(d)
    if not by:return pd.DataFrame()
    d=pd.concat(by,ignore_index=True)
    if 'view' not in d:d['view']='synthetic_one_view'
    groups=['stage','view','condition','component','alpha','window']
    # One observation per replicate/key; aggregate replicates, never images.
    if d.duplicated(groups+['rep']).any():raise ValueError('duplicate calibration replicate/key')
    g=d.groupby(groups,dropna=False)
    result=g.agg(mean=('rate','mean'),sd=('rate','std'),n=('rate','count'),
                 min_images_per_rep=('n','min'),max_images_per_rep=('n','max')).reset_index()
    result=result[result.n>0].copy()
    half=t.ppf(.975,result.n-1)*result.sd/np.sqrt(result.n)
    result['lo']=result['mean']-half;result['hi']=result['mean']+half
    result['unit']='independent regeneration' if stage=='synthetic' else 'calibration resample on a fixed cohort'
    return result.drop(columns='sd')


def main():
    rows=[];receipts=[];status={}
    for stage in ['primary','tuning','capacity','baselines']:
        paths=sorted((H/'results'/stage).glob('draw*.json'));status[stage]={'completed':len(paths),'planned':30}
        for p in paths:
            z=json.loads(p.read_text());receipts.append(dict(stage=stage,run=p.stem,**{k:z[k] for k in z if k!='metrics'}))
            for r in z['metrics']:
                r={**r,'stage':stage,'configuration':r.get('configuration','cfg0')};rows.append(r)
    for stage in ['real_calibration','synthetic']:
        status[stage]={'completed':len(list((H/'results'/stage).glob('rep*.json'))),'planned':50 if stage=='real_calibration' else 200}
    (OUT/'completion_status.json').write_text(json.dumps(status,indent=2))
    df=pd.DataFrame(rows)
    if df.empty:return
    df.to_csv(OUT/'all_run_metrics.csv',index=False)
    primary=df[df.stage=='primary'].copy();primary.to_csv(OUT/'lp_baselines.csv',index=False)
    comparisons=[]
    # All comparisons are exploratory development analyses, including selected winners.
    for suffix in ['', '+TINS']:
        pairs=[('REPRISE_L0_M0','L0_raw'),('REPRISE_L0_M0','L2_raw'),('REPRISE_L0_M0','class_L2_raw'),
               ('REPRISE_L0_M0','L0_plp'),('L0_plp','L0_norm'),('L0_plp','L0_cdf'),
               ('L1_plp','L0_plp'),('L2_plp','L0_plp'),('L2_raw','L0_raw'),('L1_raw','L0_raw'),
               ('L2_plp','L2_norm'),('L2_plp','L2_cdf'),('REPRISE_L1_M1','REPRISE_L1_M0'),('REPRISE_L2_M1','REPRISE_L2_M0'),
               ('REPRISE_L1_M0','REPRISE_L0_M0'),('REPRISE_L1_M1','REPRISE_L0_M0'),('M0_pt','static'),('M1_pt','static')]
        for a,b in pairs:
            for stream in ['near','far']:
                for metric in ['FPR95','AUROC']:
                    x=paired(primary,a+suffix,b+suffix,stream,metric)
                    if x:comparisons.append(dict(a=a+suffix,b=b+suffix,stream=stream,metric=metric,**x))
    # Additional read-only diagnostics: effect of the same cached TINS multiplier.
    for method in ['L2_raw','L2_norm','L0_plp','L1_plp','M0_pt','REPRISE_L0_M0','REPRISE_L1_M0','REPRISE_L1_M1']:
        for stream in ['near','far']:
            for metric in ['FPR95','AUROC']:
                x=paired(primary,method+'+TINS',method,stream,metric)
                if x:comparisons.append(dict(a=method+'+TINS',b=method,stream=stream,metric=metric,diagnostic='same cached TINS multiplier',**x))
    means=[]
    for (stream,method),g in primary.groupby(['stream','method']):
        if not all(g.groupby('draw').seed.nunique()==3):continue
        for metric in ['FPR95','AUROC']:
            means.append(dict(stream=stream,method=method,metric=metric,**ci(g.groupby('draw')[metric].mean().values)))
    pd.DataFrame(means).to_csv(OUT/'primary_means.csv',index=False)
    primary_cal=[]
    for p in (H/'results/primary').glob('draw*.npz'):
        if 'calibration' in p.name:continue
        if not p.with_suffix('.json').exists():continue
        draw,stream,seed=p.stem.replace('draw','').replace('seed','').split('_')
        z=np.load(p,allow_pickle=True);mask=~z['is_ood'].astype(bool)
        for view in ['B14','L14']:
            for component in ['static','pall','M0_pA1','M0_pA2','M0_pt','M1_pall','M1_pA1','M1_pA2','M1_pt','L0_plp','L1_plp','L2_plp']:
                for alpha in [.01,.05,.1]:
                    primary_cal.append(dict(draw=int(draw),seed=int(seed),stream=stream,view=view,component=component,alpha=alpha,
                                            rate=float(np.mean(z[f'{view}_{component}'][mask]<=alpha))))
    pd.DataFrame(primary_cal).to_csv(OUT/'primary_calibration_rates.csv',index=False)
    # Strongest simple LP chosen on dev1 near only, requiring all 15 paired conditions.
    selected={}
    for suffix in ['', '+TINS']:
        cand=primary[(primary.stream=='near')&primary.method.str.endswith(('raw'+suffix,'norm'+suffix,'cdf'+suffix))]
        cnt=cand.groupby('method').size();eligible=cnt[cnt==15].index
        if len(eligible):
            best=cand[cand.method.isin(eligible)].groupby('method').FPR95.mean().idxmin();selected[suffix or 'standalone']=best
            for stream in ['near','far']:
                for metric in ['FPR95','AUROC']:
                    x=paired(primary,'REPRISE_L0_M0'+suffix,best,stream,metric)
                    if x:comparisons.append(dict(a='REPRISE_L0_M0'+suffix,b=best,stream=stream,metric=metric,selection='best dev simple LP',**x))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'selected_simple_lp.json').write_text(json.dumps(selected,indent=2))
    # Separate resource-matched tuning table; cfg0 comes from the primary table.
    matched=df[df.stage.isin(['primary','tuning','capacity','baselines'])].copy()
    matched.to_csv(OUT/'matched_resource_comparison.csv',index=False)
    solver=[]
    for p in (H/'results/primary').glob('*_solver.csv'):
        d=pd.read_csv(p);d['run']=p.stem.replace('_solver','');solver.append(d)
    if solver:
        sd=pd.concat(solver,ignore_index=True);sd.to_csv(OUT/'solver_audit.csv',index=False)
        columns=[c for c in ['run','view','batch','solver','support_median','cal_q01','cal_q50','raw_median','raw_tie_fraction','p_tie_fraction','rank_reversals','boundary_ties'] if c in sd]
        sd[columns].to_csv(OUT/'rank_scale_diagnostics.csv',index=False)
    cal=[aggregate_calibration(s) for s in ['synthetic','real_calibration']]
    if any(not x.empty for x in cal):pd.concat(cal,ignore_index=True).to_csv(OUT/'calibration_audit.csv',index=False)
    real=[]
    ip=H/'data/real_calibration/identities.npz'
    if ip.exists():
        identities=np.load(ip);flag0=identities['is_ood']
        for p in (H/'results/real_calibration').glob('rep*.npz'):
            rep=int(p.stem[3:]);z=np.load(p)
            for cond in ['mixed','class_burst']:
                # Real-feature burst preserves OOD positions, so the label mask is identical.
                scores={}
                for mode in ['L0','L1','L2']:
                    lp=sum(log_nonnegative(z[f'{v}_{cond}_{mode}_plp']) for v in ['B14','L14'])
                    scores[mode+'_LP']=lp
                    for mem in ['M0','M1']:
                        scores[mode+'_'+mem]=lp+sum(log_nonnegative(z[f'{v}_{cond}_{mem}_pt']) for v in ['B14','L14'])
                for method,s in scores.items():real.append(dict(rep=rep,condition=cond,method=method,**metrics(s,flag0)))
        pd.DataFrame(real).to_csv(OUT/'real_detection_metrics.csv',index=False)
    (OUT/'runtime_receipts.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps({'status':status,'selected_simple_lp':selected,'comparisons':len(comparisons)}),flush=True)


if __name__=='__main__':main()
