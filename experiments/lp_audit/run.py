import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from pilot import load_development,prepare
from core import run_view
from metrics import metrics,log_nonnegative
HERE=Path(__file__).resolve().parent


def run(draw,stream,seed,classwise=False,n=None,tag='primary'):
    start=time.perf_counter();torch.set_num_threads(1)
    pr_path=HERE/'preregistration.json';pr_bytes=pr_path.read_bytes();pr=json.loads(pr_bytes)
    assert hashlib.sha256(pr_bytes).hexdigest()==(HERE/'preregistration.sha256').read_text().strip()
    R,D=load_development();results={};diags=[];allcal={};views={};prep_s=0
    for name in ['B14','L14']:
        t=time.perf_counter();su,ca,sf,bi,v,sid,flag,S=prepare(R,D,draw,name,stream,seed,n);prep_s+=time.perf_counter()-t
        vals,diag,cals=run_view(su,ca,sf,bi,v,tuple(pr['thresholds_exact']),device='cuda:0',classwise=classwise)
        views[name]=vals
        for d in diag:d.update(view=name)
        diags+=diag
        for k,x in cals.items():allcal[f'{name}_{k}']=x
        results.update({f'{name}_{k}':x for k,x in vals.items()})
    out=HERE/'results'/tag/f'draw{draw}_{stream}_seed{seed}'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.json').exists():raise FileExistsError(out)
    scores={}
    names=['static','M0_pt','M1_pt']+[f'{m}_{tr}' for m in ['L0','L1','L2'] for tr in ['raw','norm','cdf','plp']]
    if classwise:names += [f'class_L2_{tr}' for tr in ['raw','norm','cdf','plp']]
    for name in names:
        scores[name]=sum(log_nonnegative(views[v][name]) for v in ['B14','L14'])
    for mode,mem in [('L0','M0'),('L1','M0'),('L2','M0'),('L1','M1'),('L2','M1')]:
        scores[f'REPRISE_{mode}_{mem}']=scores[f'{mode}_plp']+scores[f'{mem}_pt']
    scores['TINS']=log_nonnegative(S)
    for k,x in list(scores.items()):
        if k!='TINS':scores[k+'+TINS']=x+scores['TINS']
    rows=[]
    for method,score in scores.items():
        met=metrics(score,flag);met.update(method=method,draw=draw,stream=stream,seed=seed)
        batch_au=[]
        for b in np.unique(bi):
            ix=bi==b
            if flag[ix].any() and (~flag[ix]).any():batch_au.append(metrics(score[ix],flag[ix])['AUROC'])
        met['within_batch_AUROC']=float(np.mean(batch_au));rows.append(met)
    results.update({f'log_score_{k}':v for k,v in scores.items()})
    results.update(sample_id=sid,is_ood=flag,batch_index=bi,S=S)
    for name in names:
        prod=views['B14'][name]*views['L14'][name]
        if np.any(prod==0) and np.all(views['B14'][name]>0) and np.all(views['L14'][name]>0):raise FloatingPointError('product underflow')
    np.savez_compressed(out.with_suffix('.npz'),**results)
    np.savez_compressed(str(out)+'_calibration.npz',**allcal)
    pd.DataFrame(diags).to_csv(str(out)+'_solver.csv',index=False)
    cpu_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    receipt={'metrics':rows,'wall_seconds':time.perf_counter()-start,'prepare_seconds':prep_s,
             'gpu_hours_reserved_conservative':(time.perf_counter()-start)/3600,
             'gpu_allocated_peak':torch.cuda.max_memory_allocated(),'gpu_reserved_peak':torch.cuda.max_memory_reserved(),
             'cpu_peak_bytes':cpu_peak,'classwise':classwise,'n_stream':len(sid),'prereg_sha256':hashlib.sha256(pr_bytes).hexdigest(),
             'code_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['core.py','run.py','metrics.py','pilot.py']}}
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'completed':str(out),'seconds':receipt['wall_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True);p.add_argument('--stream',choices=['near','far'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--classwise',action='store_true');p.add_argument('--n',type=int)
    p.add_argument('--tag',default='primary');a=p.parse_args()
    try:run(a.draw,a.stream,a.seed,a.classwise,a.n,a.tag)
    except BaseException:
        (HERE/'failures').mkdir(exist_ok=True)
        (HERE/'failures'/f'{a.tag}_{a.draw}_{a.stream}_{a.seed}_{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
