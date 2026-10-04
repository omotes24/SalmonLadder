"""Capacity-matched alternatives. Eviction applies to A1, A2, M and graph stream nodes."""
import time
import numpy as np
import torch
from core import PrefixGraph,solve,class_max_solve,p_high,p_low


def prune_graph(graph,keep):
    """Exact deletion update: only rows losing a top-k neighbor need a new search."""
    oldn=len(graph.x);mapping=np.full(oldn,-1,np.int64);mapping[keep]=np.arange(len(keep))
    indices=mapping[graph.ti[keep]];affected=np.any(indices<0,axis=1)
    graph.x=graph.x[keep];graph.global_ids=graph.global_ids[keep]
    graph.ts=graph.ts[keep];graph.ti=indices
    if graph.tx is not None:graph.tx=graph.tx[torch.as_tensor(keep,device=graph.device)]
    bad=np.flatnonzero(affected)
    for lo in range(0,len(bad),512):
        rows=bad[lo:lo+512]
        if graph.tx is None:sims=graph.x[rows]@graph.x.T
        else:sims=(graph.tx[torch.as_tensor(rows,device=graph.device)]@graph.tx.T).cpu().numpy()
        sims[np.arange(len(rows)),rows]=-np.inf
        graph.ts[rows],graph.ti[rows]=graph.top(sims)
    return graph


def bounded_view(support,cal,sf,bidx,v,thresholds,capacity=8192,device='cuda:0',classwise=True):
    ns=support.shape[0]*support.shape[1];nc=len(cal);nf=ns+nc;n=len(sf)
    fixed=np.r_[support.reshape(ns,-1),cal];graph=PrefixGraph(fixed,device=device)
    y=np.r_[np.ones(ns),np.zeros(nc)]
    modes=['L0','L1','L2'];prev={m:None for m in modes}
    frozen={m:solve(graph.matrix(),y,m)[0][ns:] for m in modes}
    classes=np.repeat(np.arange(len(support)),support.shape[1])
    frozen_class=class_max_solve(graph.matrix(),classes,ns,device=device)[0][ns:] if classwise else None
    tx=torch.as_tensor(sf,device=device);tc=torch.as_tensor(cal,device=device)
    mem={mn:[[] for _ in range(3)] for mn in ['M0','M1']}
    arr={k:np.empty(n) for k in ['M0_pt','M1_pt','static']}
    arr['static'][:]=v['p'][nc:]
    for mode in modes+(['class_L2'] if classwise else []):
        for tr in ['raw','norm','cdf','plp']:arr[f'{mode}_{tr}']=np.empty(n)
    for mn in mem:
        for j in range(3):arr[f'{mn}_admit{j}']=np.empty(n,bool)
    med,mad=v['stats']['med_all'],v['stats']['mad_all'];diag=[]
    for b in np.unique(bidx):
        rows=np.flatnonzero(bidx==b);tick=time.perf_counter()
        keep_past=capacity-len(rows)
        if len(graph.x)-nf>keep_past:
            keep=np.r_[np.arange(nf),np.arange(len(graph.x)-keep_past,len(graph.x))]
            graph=prune_graph(graph,keep)
            for m in prev:
                if prev[m] is not None:prev[m]=prev[m][keep]
        w,_,oldn=graph.append(sf[rows]);new=np.arange(oldn,w.shape[0]);graph_s=time.perf_counter()-tick
        assert w.shape[0]-nf<=capacity
        for mn in mem:
            roles=[np.arange(nc)]*4 if mn=='M0' else [np.arange(j,nc,4) for j in range(4)]
            ps=[p_high(v['d_all_cal'][roles[0]],v['d_all'][nc+rows])]
            for j in range(3):
                ids=mem[mn][j]
                if ids:
                    mt=tx[ids]
                    rs=(1-(tx[rows]@mt.T).max(1).values).cpu().numpy().astype(np.float64)
                    rc=1-(tc@mt.T).max(1).values.cpu().numpy().astype(np.float64)
                else:rs=np.full(len(rows),2.);rc=np.full(nc,2.)
                g=v['d'][nc+rows]-(rs-med)/mad;gc=v['d_cal']-(rc-med)/mad
                ps.append(p_high(gc[roles[j+1]],g))
            arr[f'{mn}_pt'][rows]=ps[-1]
            for j in range(3):
                mask=ps[j]<=thresholds[j];arr[f'{mn}_admit{j}'][rows]=mask
                mem[mn][j]=(mem[mn][j]+rows[mask].tolist())[-capacity:]
                assert len(mem[mn][j])<=capacity
        yy=np.r_[np.ones(ns),np.zeros(w.shape[0]-ns)]
        for m in modes:
            u,d=solve(w,yy,m,prev[m]);prev[m]=u.copy()
            arr[f'{m}_raw'][rows]=u[new];arr[f'{m}_norm'][rows]=u[new]/max(float(np.median(u[:ns])),1e-12)
            arr[f'{m}_cdf'][rows]=p_low(frozen[m],u[new]);arr[f'{m}_plp'][rows]=p_low(u[ns:nf],u[new])
            d.update(batch=int(b),solver=m,n_nodes=w.shape[0],graph_seconds=graph_s,capacity=capacity,
                     max_A1_A2_M=max(len(mm) for sets in mem.values() for mm in sets));diag.append(d)
        if classwise:
            u,d=class_max_solve(w,classes,ns,device=device)
            for tr,val in [('raw',u[new]),('norm',u[new]/max(float(np.median(u[:ns])),1e-12)),
                           ('cdf',p_low(frozen_class,u[new])),('plp',p_low(u[ns:nf],u[new]))]:arr[f'class_L2_{tr}'][rows]=val
            d.update(batch=int(b),solver='class_L2',n_nodes=w.shape[0],capacity=capacity);diag.append(d)
    return arr,diag


def memory_baselines(support,cal,sf,bidx,capacity=None,device='cuda:0'):
    """Matched 12-support+4-calibration AdaNeg/OODD-type implementations, not official methods."""
    mu=support.mean(1);mu/=np.linalg.norm(mu,axis=1,keepdims=True)
    sid=(sf@mu.T).max(1);cid=(cal@mu.T).max(1)
    tx=torch.as_tensor(sf,device=device);tc=torch.as_tensor(cal,device=device)
    configs={f'AdaNeg_type_cfg{i}':('ada',tau,weight,None) for i,(tau,weight) in enumerate([(.45,1.),(.55,2.),(.65,1.)])}
    configs.update({f'OODD_type_cfg{i}':('oodd',None,1.,k) for i,k in enumerate([1024,4096,8192])})
    mem={k:[] for k in configs};out={k:np.empty(len(sf)) for k in configs};traces=[]
    for b in np.unique(bidx):
        rows=np.flatnonzero(bidx==b)
        for key,(typ,tau,weight,limit) in configs.items():
            ids=mem[key]
            if ids:
                mt=tx[ids];kk=min(5,len(ids))
                qs=(tx[rows]@mt.T).topk(kk,dim=1).values.mean(1).cpu().numpy()
                cs=(tc@mt.T).topk(kk,dim=1).values.mean(1).cpu().numpy()
            else:qs=np.zeros(len(rows));cs=np.zeros(len(cal))
            sc=sid[rows]-weight*qs;cc=cid-weight*cs
            out[key][rows]=p_low(cc,sc)
            if typ=='ada':
                ids=ids+rows[sid[rows]<tau].tolist()
                if capacity:ids=ids[-capacity:]
            else:
                ids=ids+rows.tolist();maxkeep=min(limit,capacity) if capacity else limit
                if len(ids)>maxkeep:
                    ii=np.asarray(ids);chosen=np.argsort(sid[ii],kind='stable')[:maxkeep];ids=ii[chosen].tolist()
            mem[key]=ids
            traces.append(dict(batch=int(b),method=key,memory_size=len(ids),capacity=capacity))
    return out,traces
