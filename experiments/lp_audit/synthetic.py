import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import concurrent.futures as cf,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'vendor'))
from vins import r5
from core import run_view,PrefixGraph,solve


def norm(x):return (x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-15)).astype(np.float32)


def task(rep):
    cfg=json.loads((H/'preregistration.json').read_text());sp=cfg['simulation'];rng=np.random.default_rng(sp['seed_base']+rep)
    c,dim,n=sp['ID_classes'],sp['dimension'],sp['stream_length']
    mu=norm(rng.normal(size=(c+2,dim)))
    sup=norm(mu[:c,None,:]+rng.normal(0,.3,(c,12,dim)))
    iid_labels=rng.integers(c,size=c*4);balanced_labels=np.repeat(np.arange(c),4)
    noise=rng.normal(0,.3,(c*4,dim))
    nc=c*4;bi=np.arange(n)//16
    rows=[];start=time.perf_counter();raw={}
    for condition in sp['conditions']:
        cl=balanced_labels if condition=='class_balanced_cal_mixed' else iid_labels
        cal=norm(mu[cl]+noise)
        stream_cls=rng.integers(c,size=n)
        flag=np.zeros(n,bool)
        if condition!='iid_ID_only':flag[rng.choice(n,n//4,replace=False)]=True;stream_cls[flag]=rng.integers(c,c+2,size=flag.sum())
        sf0=mu[stream_cls]+rng.normal(0,.3,(n,dim))
        if condition=='ID_shift_mixed':sf0[~flag]+= .8*mu[-1]
        sf=norm(sf0)
        if condition=='class_burst_mixed':
            order=np.argsort(stream_cls,kind='stable');sf,flag,stream_cls=sf[order],flag[order],stream_cls[order]
        q=np.r_[cal,sf];v=r5.proto_view(sup,q,np.tile(np.arange(c),(len(q),1)),np.arange(len(q))<nc,n0=48,m=1)
        a,diags,_=run_view(sup,cal,sf,bi,v,tuple(cfg['thresholds_exact']),device='cpu')
        for key in ['static','pall','M0_pA1','M0_pA2','M0_pt','M1_pall','M1_pA1','M1_pA2','M1_pt','L0_plp','L1_plp','L2_plp']:
            raw[f'{condition}_{key}']=a[key]
            for alpha in [.01,.05,.1]:
                for window,mask in [('all',~flag),('early',(~flag)&(bi<4)),('late',(~flag)&(bi>=4))]:
                    rows.append(dict(rep=rep,condition=condition,component=key,alpha=alpha,window=window,
                                     rate=float(np.mean(a[key][mask]<=alpha)) if mask.any() else np.nan,n=int(mask.sum())))
                for cls in range(c):
                    mask=(~flag)&(stream_cls==cls)
                    if mask.any():rows.append(dict(rep=rep,condition=condition,component=key,alpha=alpha,window=f'class{cls}',rate=float(np.mean(a[key][mask]<=alpha)),n=int(mask.sum())))
        for key in ['M0_admit2','M1_admit2']:
            for category,mask in [('ID',~flag),('OOD',flag)]:
                rows.append(dict(rep=rep,condition=condition,component=key,alpha=cfg['thresholds_exact'][2],window=category,
                                 rate=float(np.mean(a[key][mask])) if mask.any() else np.nan,n=int(mask.sum())))
        raw[f'{condition}_is_ood']=flag;raw[f'{condition}_class']=stream_cls
    out=H/'results/synthetic';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'rep{rep}.npz',**raw)
    pd.DataFrame(rows).to_csv(out/f'rep{rep}.csv',index=False)
    (out/f'rep{rep}.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'seed':sp['seed_base']+rep,'gpu_seconds':0}))
    return rep


def symmetry_diagnostics():
    rng=np.random.default_rng(555);sup=norm(rng.normal(size=(24,8)));cal=norm(rng.normal(size=(12,8)));sf=norm(rng.normal(size=(32,8)))
    def replay(c,s):
        graph=PrefixGraph(np.r_[sup,c],k=5);prev={m:None for m in ['L0','L1','L2']};out={}
        for lo in range(0,len(s),8):
            w,_,_=graph.append(s[lo:lo+8]);y=np.r_[np.ones(24),np.zeros(len(graph.x)-24)]
            for mode in prev:
                u,d=solve(w,y,mode,prev[mode]);prev[mode]=u.copy();out[mode]=u
        return out
    a=replay(cal,sf);ca=cal.copy();ss=sf.copy();ca[0],ss[-1]=sf[-1].copy(),cal[0].copy();b=replay(ca,ss)
    p=np.arange(68);p[24],p[-1]=p[-1],p[24]
    result={'role_swap_max_abs':{m:float(np.max(np.abs(a[m]-b[m][p]))) for m in a},
            'meaning':'Full history replay after exchanging one persistent calibration and one final arriving ID. L0 age asymmetry is measured, not inferred from finite iterations alone.'}
    # Exact duplicate similarities stress array-index-dependent kNN ties.
    x=np.repeat(norm(rng.normal(size=(3,8))),12,axis=0);perm=rng.permutation(len(x));g=PrefixGraph(x,k=5);gp=PrefixGraph(x[perm],k=5)
    w=g.matrix().toarray();wp=gp.matrix().toarray()[np.argsort(perm)][:,np.argsort(perm)]
    result['tied_graph_permutation_max_abs']=float(np.max(np.abs(w-wp)))
    (H/'reports/symmetry_diagnostics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    symmetry_diagnostics()
    n=json.loads((H/'preregistration.json').read_text())['simulation']['independent_replicates']
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(task,range(n)):
            if r%10==0:print(json.dumps({'complete':r}),flush=True)
