import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from core import run_view
from vins import r5


def load_development():
    spec=importlib.util.spec_from_file_location('round2',HERE/'vendor/scripts/r5_round2.py')
    R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    return R,R.load_dev('dev1')


def prepare(R,D,draw,name,stream,seed,n=None):
    v,sup,q,nc=R.view_inputs(D,str(draw),name,48,20,1)
    sid,flag,S,bidx=D['tins'][(str(draw),stream,seed)]
    if n:sid,flag,S,bidx=sid[:n],flag[:n],S[:n],bidx[:n]
    rows=np.array([D['stream_row'][s] for s in sid]);qi=nc+rows
    vv={**v}
    for key in ['d','d_all','p','p_all']:vv[key]=np.r_[v[key][:nc],v[key][qi]]
    return sup,q[:nc],q[qi],bidx,vv,sid,flag,S


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=512);p.add_argument('--classwise',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);start=time.perf_counter();R,D=load_development();load_s=time.perf_counter()-start
    su,ca,sf,bi,v,*_=prepare(R,D,0,'L14','near',123,a.n)
    start=time.perf_counter();arr,diag,_=run_view(su,ca,sf,bi,v,(.3,.2,.10191613435745239),a.device,a.classwise)
    wall=time.perf_counter()-start
    out={'stage':'cost_pilot_not_selection','n':a.n,'load_seconds':load_s,'wall_seconds':wall,'classwise':a.classwise,
         'n_support':len(su)*12,'n_calibration':len(ca),'diag':diag,
         'gpu_allocated':torch.cuda.max_memory_allocated(),'gpu_reserved':torch.cuda.max_memory_reserved()}
    if not a.classwise:
        start=time.perf_counter();ref=r5.lp_run(su,ca,sf,bi,k=10,alpha=.9,gamma=1.,iters=15)
        out['legacy_cpu_seconds']=time.perf_counter()-start
        out['legacy_max_u_error']=float(np.max(np.abs(arr['L0_raw']-ref['u'])))
        out['legacy_max_p_error']=float(np.max(np.abs(arr['L0_plp']-ref['p'])))
        out['legacy_changed_p_count']=int(np.sum(arr['L0_plp']!=ref['p']))
    path=HERE/'reports'/f'pilot_{a.n}_class{int(a.classwise)}.json';path.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='diag'}),flush=True)


if __name__=='__main__':main()
