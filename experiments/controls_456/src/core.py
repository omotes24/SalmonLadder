"""REPRISE control experiments. Labels are absent from every detector/state API.

Float32 exact cosine searches (no approximate index); float64 propagation/calibration.
The finite-window rule keeps the last W stream arrivals in each of A1/A2/M and graph.
Support and calibration are permanent. Probes clone state, insert once, and discard it.
"""
import copy
import math
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

def p_high(cal, x):
    c=np.sort(np.asarray(cal,dtype=np.float64))
    return (1+len(c)-np.searchsorted(c,x,side='left'))/(len(c)+1)

def p_low(cal,x):
    c=np.sort(np.asarray(cal,dtype=np.float64))
    return (1+np.searchsorted(c,x,side='right'))/(len(c)+1)

def norm(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)

def static_view(support, cal, query, cand_cal, cand_query):
    """Frozen v4 prototype statistics; memory scale still uses support 2-NN LOO."""
    S=np.asarray(support,dtype=np.float64)
    tot=S.sum(1); mu=tot/np.linalg.norm(tot,axis=1,keepdims=True)
    lm=tot[:,None,:]-S; lm/=np.linalg.norm(lm,axis=2,keepdims=True)
    dist=1-np.einsum('cnd,cnd->cn',S,lm)
    med=np.median(dist,axis=1); mad=1.4826*np.median(abs(dist-med[:,None]),axis=1)
    gm=float(np.median(dist)); gs=float(1.4826*np.median(abs(dist-gm)))
    mt=(12*med+12*gm)/24; st=(12*mad+12*gs)/24
    if np.any(st<=0): raise ValueError('nonpositive prototype scale')
    dd=1-np.einsum('cnd,cmd->cnm',S,S)
    for c in range(len(S)):np.fill_diagonal(dd[c],np.inf)
    loo=np.sort(dd,axis=2)[:,:,1]
    mm=float(np.median(loo)); ms=float(1.4826*np.median(abs(loo-mm)))
    if ms<=0: raise ValueError('nonpositive memory scale')
    def calc(X,K):
        ds=[]; das=[]
        for lo in range(0,len(X),1024):
            z=(1-np.asarray(X[lo:lo+1024],dtype=np.float64)@mu.T-mt)/st
            ds.append(np.take_along_axis(z,K[lo:lo+1024],axis=1).min(1));das.append(z.min(1))
        return np.concatenate(ds),np.concatenate(das)
    dc,dca=calc(cal,cand_cal); d,da=calc(query,cand_query)
    return {'support':S.astype(np.float32),'cal':np.asarray(cal,dtype=np.float32),
            'dcal':dc,'dallcal':dca,'d':d,'p':p_high(dc,d),'pall':p_high(dca,da),'med':mm,'mad':ms,'mu':mu,'mt':mt,'st':st}

def view_queries(v,features,candidates):
    z=(1-np.asarray(features,dtype=np.float64)@v['mu'].T-v['mt'])/v['st']
    d=np.take_along_axis(z,candidates,axis=1).min(1);da=z.min(1)
    return d,p_high(v['dcal'],d),p_high(v['dallcal'],da)

class Graph:
    def __init__(self,support,cal,device='cpu',cap=None,mode='warm15',cache=None):
        self.device=torch.device(device); self.cap=cap; self.mode=mode
        self.ns=len(support); self.nc=len(cal); self.nfix=self.ns+self.nc
        self.X=torch.as_tensor(np.concatenate([support,cal]),device=self.device)
        self.k=10; self.alpha=.9; self.gamma=3.
        self.u=np.r_[np.ones(self.ns),np.zeros(self.nc)]
        self.steps=0; self.last={}
        if cache is not None and Path(cache).exists():
            z=np.load(cache); self.s=torch.as_tensor(z['s'],device=self.device); self.i=torch.as_tensor(z['i'],device=self.device)
            assert self.s.shape==(self.nfix,self.k)
        else:
            self.s,self.i=self._neighbors(self.X,torch.arange(self.nfix,device=self.device))
            if cache is not None:
                Path(cache).parent.mkdir(parents=True,exist_ok=True)
                import os
                tmp=str(cache)+f'.{os.getpid()}.npz'
                np.savez(tmp,s=self.s.cpu().numpy(),i=self.i.cpu().numpy());os.replace(tmp,cache)

    def clone(self):
        g=copy.copy(self)
        g.X=self.X.clone();g.s=self.s.clone();g.i=self.i.clone();g.u=self.u.copy();g.last=dict(self.last)
        return g

    def _neighbors(self,Q,self_indices=None):
        ss=[];ii=[]
        for lo in range(0,len(Q),512):
            z=Q[lo:lo+512]@self.X.T
            if self_indices is not None:
                z[torch.arange(len(z),device=self.device),self_indices[lo:lo+512]]=-torch.inf
            vals,inds=torch.topk(z,self.k,dim=1,sorted=True)
            ss.append(vals);ii.append(inds)
        return torch.cat(ss),torch.cat(ii)

    def _prune(self,nadd):
        if self.cap is None:return 0
        drop=max(0,len(self.X)-self.nfix+nadd-self.cap)
        if not drop:return 0
        keep=torch.cat([torch.arange(self.nfix,device=self.device),torch.arange(self.nfix+drop,len(self.X),device=self.device)])
        mapping=torch.full((len(self.X),),-1,device=self.device,dtype=torch.long);mapping[keep]=torch.arange(len(keep),device=self.device)
        ni=mapping[self.i[keep]];bad=(ni<0).any(1)
        self.X=self.X[keep];self.s=self.s[keep];self.i=ni;self.u=self.u[keep.cpu().numpy()]
        if bool(bad.any()):
            ids=torch.where(bad)[0]
            self.s[ids],self.i[ids]=self._neighbors(self.X[ids],ids)
        return int(drop)

    def append(self,q):
        t=time.perf_counter();q=torch.as_tensor(q,device=self.device,dtype=torch.float32)
        if self.cap is not None and len(q)>self.cap:raise ValueError('batch exceeds cap')
        removed=self._prune(len(q));old=len(self.X)
        self.X=torch.cat([self.X,q]);n=len(self.X)
        sim=q@self.X.T
        sim[torch.arange(len(q),device=self.device),torch.arange(old,n,device=self.device)]=-torch.inf
        ns,ni=torch.topk(sim,self.k,dim=1,sorted=True)
        v=torch.cat([self.s,sim[:,:old].T],1)
        idx=torch.cat([self.i,torch.arange(old,n,device=self.device).expand(old,-1)],1)
        val,at=torch.topk(v,self.k,dim=1,sorted=True)
        self.s=torch.cat([val,ns]);self.i=torch.cat([torch.gather(idx,1,at),ni])
        self.u=np.r_[self.u,np.zeros(len(q))]
        if self.device.type=='cuda':torch.cuda.synchronize()
        t1=time.perf_counter()
        ss=self.s.cpu().numpy();ii=self.i.cpu().numpy()
        a=sp.csr_matrix((np.maximum(ss.ravel(),0).astype(np.float32)**3,(np.repeat(np.arange(n),10),ii.ravel())),shape=(n,n))
        W=a.maximum(a.T);deg=np.asarray(W.sum(1)).ravel();inv=1/np.sqrt(np.maximum(deg,1e-12))
        Wn=sp.diags(inv)@W@sp.diags(inv)
        t2=time.perf_counter()
        y=np.zeros(n);y[:self.ns]=.1
        u=self.u if self.mode=='warm15' else np.zeros(n)
        residual=None;converged=self.mode=='warm15'
        limit=15 if self.mode=='warm15' else 500
        for it in range(limit):
            nxt=.9*(Wn@u)+y
            residual=np.linalg.norm(nxt-u)/(.1*max(np.linalg.norm(nxt),1e-30))
            u=nxt
            if self.mode=='converged' and residual<=1e-6:
                converged=True;break
        if not converged:raise RuntimeError(f'LP convergence failed, residual={residual}')
        self.u=u;self.steps+=1
        self.last={'nearest_s':t1-t,'graph_build_s':t2-t1,'propagation_s':time.perf_counter()-t2,
                   'sweeps':it+1,'relative_error_bound':float(residual),'nodes':n,'stream_nodes':n-self.nfix,'evicted':removed}
        return p_low(u[self.ns:self.nfix],u[old:]),u[old:]

class Detector:
    def __init__(self,view,device='cpu',cap=None,mode='warm15',cache=None):
        self.v=view;self.device=torch.device(device);self.cap=cap;self.mode=mode
        self.cal=torch.as_tensor(view['cal'],device=self.device)
        dim=self.cal.shape[1]
        self.banks=[torch.empty((0,dim),device=self.device) for _ in range(3)]
        self.ages=[np.zeros(0,dtype=np.int64) for _ in range(3)]
        self.seen=0
        self.graph=Graph(view['support'].reshape(-1,dim),view['cal'],device,cap,mode,cache)
        self.last={}

    def clone(self):
        d=copy.copy(self);d.graph=self.graph.clone()
        # read-only banks are replaced, never edited in-place.
        d.banks=list(self.banks);d.ages=[a.copy() for a in self.ages];d.last=dict(self.last)
        return d

    def _memory_p(self,q,d,bank):
        def dist(Q):
            if len(bank)<2:return np.full(len(Q),2.)
            arr=[]
            for lo in range(0,len(Q),512):
                arr.append(1-torch.topk(Q[lo:lo+512]@bank.T,2,dim=1).values[:,1].cpu().numpy().astype(np.float64))
            return np.concatenate(arr)
        rq,rc=dist(q),dist(self.cal)
        g=np.asarray(d)-(rq-self.v['med'])/self.v['mad']
        gc=self.v['dcal']-(rc-self.v['med'])/self.v['mad']
        return p_high(gc,g)

    def step(self,features,d,pall,update=True):
        q=torch.as_tensor(features,device=self.device,dtype=torch.float32)
        t=time.perf_counter()
        p1=self._memory_p(q,d,self.banks[0]);p2=self._memory_p(q,d,self.banks[1]);pt=self._memory_p(q,d,self.banks[2])
        masks=[np.asarray(pall)<=.4,p1<=.3,p2<=.1]
        if self.device.type=='cuda':torch.cuda.synchronize()
        tm=time.perf_counter()-t
        if not update:raise ValueError('Use cloned detector for probes, to include exact graph insertion')
        plp,_=self.graph.append(q)
        inds=np.arange(self.seen,self.seen+len(q));self.seen+=len(q)
        for i,mask in enumerate(masks):
            b=torch.cat([self.banks[i],q[torch.as_tensor(mask,device=self.device)]])
            a=np.r_[self.ages[i],inds[mask]]
            keep=np.ones(len(a),bool) if self.cap is None else a>=self.seen-self.cap
            self.banks[i]=b[torch.as_tensor(keep,device=self.device)];self.ages[i]=a[keep]
        self.last={'memory_s':tm,**self.graph.last,'bank_sizes':[len(x) for x in self.banks]}
        return {'memory':pt,'lp':plp,'full':pt*plp,'admit_M':masks[2]}

def score_methods(detectors,features,views,indices,probe=False):
    out={k:np.ones(len(indices),dtype=np.float64) for k in ['static','memory','lp','full']}
    info={}
    for name,det in detectors.items():
        d=det.clone() if probe else det
        v=views[name];rr=d.step(features[name][indices],v['d'][indices],v['pall'][indices])
        out['static']*=v['p'][indices]
        for k in ['memory','lp','full']:out[k]*=rr[k]
        info[name]=dict(d.last)
    return out,info

def metrics(id_scores,ood_scores):
    from sklearn.metrics import roc_auc_score
    I=np.asarray(id_scores);O=np.asarray(ood_scores)
    if len(I)==0 or len(O)==0:return {'AUROC':None,'FPR95':None}
    if not np.isfinite(np.r_[I,O]).all():raise ValueError('nonfinite score')
    threshold=np.sort(I)[int(math.floor(.05*len(I)))]
    return {'AUROC':100*float(roc_auc_score(np.r_[np.ones(len(I)),np.zeros(len(O))],np.r_[I,O])),
            'FPR95':100*float(np.mean(O>=threshold)), 'threshold':float(threshold),
            'ID_TPR':float(np.mean(I>=threshold)), 'ID_ties':int(np.sum(I==threshold)), 'OOD_ties':int(np.sum(O==threshold))}

def operating(scores,is_ood,classes,thresholds):
    result={};o=np.asarray(is_ood,bool);cl=np.asarray(classes)
    for name,s in scores.items():
        flag=np.asarray(s)<thresholds[name]
        delays=[];missed=0;details=[]
        for c in np.unique(cl[o]):
            at=np.flatnonzero(o&(cl==c));hits=at[flag[at]]
            if len(hits):
                delay=int(hits[0]-at[0]); event=int(np.searchsorted(at,hits[0]));delays.append(delay)
            else:delay=None;event=None;missed+=1
            details.append({'class':str(c),'n':len(at),'first_index':int(at[0]),'delay_images':delay,'delay_occurrences':event,'missed':not len(hits)})
        result[name]={'ID_false_alarms_per_1000':1000*float(flag[~o].mean()) if (~o).any() else None,
                      'OOD_detection_rate':float(flag[o].mean()) if o.any() else None,
                      'undetected_class_fraction':missed/len(details) if details else None,
                      'delay_images_detected_median':float(np.median(delays)) if delays else None,
                      'class_delays':details,'threshold':thresholds[name]}
    return result
