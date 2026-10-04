import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,sys,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from pilot import load_development,prepare
from core import run_view,p_high
from bounded import bounded_view,memory_baselines
from metrics import metrics,log_nonnegative
from vins import r5
H=Path(__file__).resolve().parent


def main(draw,stream,seed,kind):
    torch.set_num_threads(1);start=time.perf_counter();R,D=load_development();cfg=json.loads((H/'preregistration.json').read_text())
    bundles={};diags=[]
    for name in ['B14','L14']:
        su,ca,sf,bi,v,sid,flag,S=prepare(R,D,draw,name,stream,seed)
        nc=len(ca)
        if kind=='tuning':
            for i,(k,alpha) in enumerate([(10,.8),(20,.9)],start=1):
                a,d,_=run_view(su,ca,sf,bi,v,tuple(cfg['thresholds_exact']),device='cuda:0',classwise=True,k=k,alpha=alpha)
                bundles.setdefault(f'cfg{i}',{})[name]=a
                for x in d:x.update(view=name,configuration=i)
                diags+=d
        elif kind=='capacity':
            a,d=bounded_view(su,ca,sf,bi,v,tuple(cfg['thresholds_exact']),capacity=8192)
            bundles.setdefault('cap8192',{})[name]=a
            for x in d:x.update(view=name)
            diags+=d
        elif kind=='baselines':
            for cap in [None,8192]:
                a,d=memory_baselines(su,ca,sf,bi,capacity=cap)
                bundles.setdefault('all' if cap is None else 'cap8192',{})[name]=a
                for x in d:x.update(view=name)
                diags+=d
            md=r5.maha_pp(su,np.r_[ca,sf],[.01,.5,.9])
            for cap in ['all','cap8192']:
                for i,lam in enumerate([.01,.5,.9]):bundles[cap][name][f'MahaPP_type_cfg{i}']=p_high(md[lam][:nc],md[lam][nc:])
    rows=[];raw={'sample_id':sid,'is_ood':flag,'batch_index':bi,'S':S}
    for setting,views in bundles.items():
        keys=[k for k in views['B14'] if 'admit' not in k and k not in ['pall','M0_pA1','M0_pA2','M1_pA1','M1_pA2','M1_pall']]
        scores={k:sum(log_nonnegative(views[vn][k]) for vn in ['B14','L14']) for k in keys}
        if kind!='baselines':
            for mode,mem in [('L0','M0'),('L1','M0'),('L2','M0'),('L1','M1'),('L2','M1')]:scores[f'REPRISE_{mode}_{mem}']=scores[f'{mode}_plp']+scores[f'{mem}_pt']
        for k,x in list(scores.items()):scores[k+'+TINS']=x+log_nonnegative(S)
        for method,s in scores.items():
            met=metrics(s,flag);met.update(method=method,configuration=setting,draw=draw,stream=stream,seed=seed,kind=kind);rows.append(met)
            raw[f'{setting}_{method}']=s
    out=H/f'results/{kind}/draw{draw}_{stream}_seed{seed}';out.parent.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.json').exists():raise FileExistsError(out)
    np.savez_compressed(out.with_suffix('.npz'),**raw);pd.DataFrame(diags).to_csv(str(out)+'_trace.csv',index=False)
    out.with_suffix('.json').write_text(json.dumps({'metrics':rows,'seconds':time.perf_counter()-start,
               'gpu_hours_reserved_conservative':(time.perf_counter()-start)/3600,'gpu_reserved_peak':torch.cuda.max_memory_reserved(),
               'gpu_allocated_peak':torch.cuda.max_memory_allocated()},indent=2))
    print(json.dumps({'complete':str(out),'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--draw',type=int);ap.add_argument('--stream');ap.add_argument('--seed',type=int)
    ap.add_argument('--kind',choices=['tuning','capacity','baselines']);a=ap.parse_args()
    try:main(a.draw,a.stream,a.seed,a.kind)
    except BaseException:
        (H/'failures').mkdir(exist_ok=True);(H/'failures'/f'{a.kind}_{a.draw}_{a.stream}_{a.seed}_{time.time_ns()}.txt').write_text(traceback.format_exc());raise
