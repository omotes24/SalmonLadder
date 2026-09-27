"""Incremental reports/figures from completed jobs only. No method/threshold selection."""
import argparse,json,time
from collections import defaultdict
import numpy as np
from data import ROOT,dump
from core import metrics

def ci(values):
    v=np.asarray(values,float)
    if len(v)<2:return [None,None]
    r=np.random.default_rng(2026092708)
    z=v[r.integers(0,len(v),(2000,len(v)))].mean(1)
    return [float(x) for x in np.quantile(z,[.025,.975])]

def cleanmean(v):
    v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else None

def collect():
    rows=[]
    for p in sorted((ROOT/'results').glob('*.json')):
        try:rows.append(json.loads(p.read_text()))
        except Exception:continue
    return rows

def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=collect();out=ROOT/'reports';out.mkdir(exist_ok=True)
    curves=defaultdict(list);low=[];hold=defaultdict(list);perf=[]
    thresholds=json.loads((ROOT/'thresholds.json').read_text())['visual'] if (ROOT/'thresholds.json').exists() else None
    for row in rows:
        t=row['task'];kind=t['kind']
        if kind=='recurrence':
            z=np.load(ROOT/'results'/f'{t["id"]}.npz')
            for cond,ms in row['metrics'].items():
                for method,v in ms.items():
                    alarms=v.get('ID_false_alarms_per_1000_fixed_dev')
                    if alarms is None and thresholds:
                        vals=z[cond+'__'+method];ni=row['n_ID_eval'];alarms=1000*float(np.mean(vals[:ni]<thresholds[method]))
                    curves[(t['dataset'],t['batch'],t['solver'],cond,t['r'],method)].append({'class':t['target_class'],'rep':t['rep'],
                            'AUROC':v['AUROC'],'FPR95':v['FPR95'],'ID_false_alarms_per_1000':alarms})
        elif kind=='stream':
            low.append({'task':t,'operating':row.get('operating_fixed_dev',{}),'metrics':row['metrics']})
        elif kind=='holdout':hold[(t['bank'],'-'.join(t['models']))].append(row)
        elif kind=='runtime':
            p={k:v for k,v in row['timing'].items() if k!='batch_timing'}
            comps=defaultdict(float)
            for bt in row['timing']['batch_timing']:
                comps['features_s']+=sum(bt['features'].values());comps['TINS_s']+=bt['TINS_s']
                for d in bt['visual'].values():
                    for c in ['memory_s','nearest_s','graph_build_s','propagation_s']:comps[c]+=d[c]
            perf.append({'task':t,**p,'component_totals_s':dict(comps)})
    curve=[]
    for key,vals in curves.items():
        d,b,s,c,r,m=key;g=defaultdict(list)
        for v in vals:g[v['class']].append(v)
        item={'dataset':d,'batch':b,'solver':s,'condition':c,'r':r,'method':m,'n_classes':len(g),'n_contexts':len(vals)}
        for measure in ['AUROC','FPR95','ID_false_alarms_per_1000']:
            x=[cleanmean([z[measure] for z in vv]) for vv in g.values()];x=[a for a in x if a is not None]
            item[measure]=cleanmean(x);item[measure+'_class_bootstrap_CI']=ci(x)
        curve.append(item)
    # Paired recurrence contrasts: full-vs-LP gain at r minus the corresponding r=0 gain.
    contrasts=[]
    for ds in ['ninco','ssb_hard']:
        for batch in [1,16,256]:
            for solver in ['warm15','converged']:
                for r in [1,2,5,10,20]:
                    maps=[]
                    for rr,m in [(r,'full'),(r,'lp'),(0,'full'),(0,'lp')]:
                        maps.append({(v['class'],v['rep']):v for v in curves.get((ds,batch,solver,'singleton',rr,m),[])})
                    common=set(maps[0]).intersection(*[set(d) for d in maps[1:]])
                    if not common:continue
                    cls=defaultdict(list);raw_full=defaultdict(list);raw_lp=defaultdict(list)
                    for k in common:
                        a,b,c,d=[mp[k]['AUROC'] for mp in maps];cls[k[0]].append((a-b)-(c-d));raw_full[k[0]].append(a-c);raw_lp[k[0]].append(b-d)
                    x=[np.mean(v) for v in cls.values()]
                    contrasts.append({'dataset':ds,'batch':batch,'solver':solver,'r':r,'n_classes':len(cls),
                                      'full_recurrence_AUROC_delta':float(np.mean([np.mean(v) for v in raw_full.values()])),
                                      'LP_recurrence_AUROC_delta':float(np.mean([np.mean(v) for v in raw_lp.values()])),
                                      'difference_in_differences_AUROC':float(np.mean(x)),'class_bootstrap_CI':ci(x)})
    for ds in ['ninco','ssb_hard']:
        for measure in ['AUROC','FPR95','ID_false_alarms_per_1000']:
            fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
            used=False
            for i,s in enumerate(['warm15','converged']):
                for j,b in enumerate([1,16,256]):
                    ax=axs[i,j]
                    for method in ['static','memory','lp','full']:
                        vv=sorted([x for x in curve if x['dataset']==ds and x['batch']==b and x['solver']==s and x['condition']=='singleton' and x['method']==method and x[measure] is not None],key=lambda x:x['r'])
                        if vv:
                            used=True;x=[v['r'] for v in vv];y=[v[measure] for v in vv];ax.plot(x,y,'o-',label=method)
                            lo=[v[measure+'_class_bootstrap_CI'][0] for v in vv];hi=[v[measure+'_class_bootstrap_CI'][1] for v in vv]
                            if all(z is not None for z in lo):ax.fill_between(x,lo,hi,alpha=.12)
                    ax.set_title(f'history batch={b}, LP={s}');ax.set_xticks([0,1,2,5,10,20]);ax.grid(alpha=.2)
                    if i==1:ax.set_xlabel('Previous same-class images')
                    if j==0:ax.set_ylabel(measure)
            if used:
                axs[0,0].legend();fig.suptitle(f'{ds}: fixed history N=1024, ID=768, OOD=256; each query restored independently')
                fig.tight_layout();fig.savefig(out/f'recurrence_{ds}_{measure}.pdf');fig.savefig(out/f'recurrence_{ds}_{measure}.png',dpi=150)
            plt.close(fig)
    # Peer effects are paired within fixed r and context; targets never act as each other's peers.
    peer=[]
    for row in rows:
        if row['task']['kind']!='recurrence':continue
        ms=row['metrics']
        if 'peer_same_class' in ms and 'peer_unrelated' in ms:
            peer.append({'task':row['task'],'delta_same_minus_unrelated':{m:{k:ms['peer_same_class'][m][k]-ms['peer_unrelated'][m][k] for k in ['AUROC','FPR95']} for m in ms['peer_same_class']}})
    if low:
        for measure in ['OOD_detection_rate','ID_false_alarms_per_1000','undetected_class_fraction']:
            fig,axs=plt.subplots(3,3,figsize=(14,11))
            for i,pattern in enumerate(['random','burst','long_gap']):
                for j,r in enumerate([1,5,20]):
                    ax=axs[i,j]
                    for cap in [1000,5000,20000,None]:
                        for m,style in [('full','-'),('lp','--')]:
                            xx=[];yy=[]
                            for p in [.001,.01,.05,.2]:
                                vs=[z['operating'].get(m,{}).get(measure) for z in low if z['task']['cap']==cap and z['task']['pattern']==pattern and z['task']['repeats']==r and z['task']['fraction']==p]
                                val=cleanmean(vs)
                                if val is not None:xx.append(p*100);yy.append(val)
                            if xx:ax.plot(xx,yy,style,marker='o',label=f'{m}, W={cap or "all"}')
                    ax.set_xscale('log');ax.set_title(f'{pattern}, repeats={r}');ax.grid(alpha=.2)
                    if i==2:ax.set_xlabel('OOD percent')
                    if j==0:ax.set_ylabel(measure)
            axs[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'operating_{measure}.pdf');fig.savefig(out/f'operating_{measure}.png',dpi=140);plt.close(fig)
    holdout=[]
    for (bank,models),rr in hold.items():
        methods=set.intersection(*[set(v['metrics']) for v in rr])
        holdout.append({'dataset':bank,'models':models,'n_orders':len(rr),
                        'metrics':{m:{k:cleanmean([v['metrics'][m][k] for v in rr]) for k in ['AUROC','FPR95']} for m in sorted(methods)}})
    failures=[json.loads(p.read_text()) for p in (ROOT/'failures').glob('*.json')]
    total=len(json.loads((ROOT/'plans/tasks.json').read_text())) if (ROOT/'plans/tasks.json').exists() else None
    result={'completed_experiment_jobs':sum(r['task']['kind']!='calibration' for r in rows),'planned_experiment_jobs':total,
            'recurrence_curves':curve,'paired_recurrence_contrasts':contrasts,'peer_effects':peer,'operating_conditions':low,
            'holdout':holdout,'runtime':perf,'failures':failures,
            'CI_scope':'resampling target classes of fixed scored histories; not a full adaptive-stream bootstrap'}
    dump(out/'summary.json',result)
    lines=['# REPRISE controls ④–⑥','',f'Completed: {result["completed_experiment_jobs"]} / {total}. Failures: {len(failures)}.','',
           'Results are partial until every condition is complete. Missing conditions are not zeros.','',
           '## Unused evaluation datasets','', '| Dataset | Features | Method | Orders | AUROC | FPR95 |','|---|---|---|---:|---:|---:|']
    for h in holdout:
        for m,v in h['metrics'].items():lines.append(f'| {h["dataset"]} | {h["models"]} | {m} | {h["n_orders"]} | {v["AUROC"]:.2f} | {v["FPR95"]:.2f} |')
    lines+=['','## Failed conditions','']+[f'- {x["job"]}: see {x["log"]}' for x in failures]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    tasks=json.loads((ROOT/'plans/tasks.json').read_text()) if total is not None else []
    failed_experiments=sum((ROOT/'failures'/f'{t["id"]}.json').exists() for t in tasks)
    return result['completed_experiment_jobs'],failed_experiments,total

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();prev=None
    while True:
        sig=(len(list((ROOT/'results').glob('*.json'))),len(list((ROOT/'failures').glob('*.json'))), (ROOT/'plans/tasks.json').exists(),(ROOT/'thresholds.json').exists())
        if sig!=prev:
            n,fail,total=build();print(json.dumps({'completed':n,'failures':fail,'total':total}),flush=True);prev=sig
            if total is not None and n+fail>=total:return
        if not a.watch:return
        time.sleep(60)

if __name__=='__main__':main()
