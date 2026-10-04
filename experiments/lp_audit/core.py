"""Causal streaming diagnostics. Detector interfaces accept no evaluation labels.

All binary LP variants share exactly the same prefix graph. Cached future features
are inaccessible to the graph until append(). Issued scores are copied out.
"""
import hashlib
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def p_low(cal, x):
    return (1 + np.searchsorted(np.sort(cal), x, side='right')) / (len(cal) + 1)


def p_high(cal, x):
    return (1 + len(cal) - np.searchsorted(np.sort(cal), x, side='left')) / (len(cal) + 1)


def normalized_graph(ts, ti, gamma=1.):
    n, k = ts.shape
    weights = np.maximum(ts.ravel(), 0.) ** gamma
    a = sp.csr_matrix((weights, (np.repeat(np.arange(n), k), ti.ravel())), shape=(n, n))
    w = a.maximum(a.T)
    deg = np.asarray(w.sum(1)).ravel()
    di = 1 / np.sqrt(np.maximum(deg, 1e-12))
    return (sp.diags(di) @ w @ sp.diags(di)).tocsr()


class PrefixGraph:
    def __init__(self, fixed, k=10, gamma=1., device='cpu', capacity=None):
        self.x = np.ascontiguousarray(fixed, dtype=np.float32)
        self.nfix = len(fixed)
        self.k, self.gamma, self.device, self.capacity = k, gamma, device, capacity
        self.global_ids = np.arange(len(fixed))
        self.next_id = len(fixed)
        self.boundary_ties = 0
        self.tx = None
        if device != 'cpu':
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            self.tx = torch.as_tensor(self.x, device=device)
        self.ts = np.empty((len(fixed), k), np.float32)
        self.ti = np.empty((len(fixed), k), np.int64)
        for lo in range(0, len(fixed), 512):
            hi = min(lo + 512, len(fixed))
            sims = self.dot_rows(lo, hi)
            sims[np.arange(hi-lo), np.arange(lo, hi)] = -np.inf
            self.ts[lo:hi], self.ti[lo:hi] = self.top(sims)

    def dot_rows(self, lo, hi):
        if self.tx is None:
            return self.x[lo:hi] @ self.x.T
        return (self.tx[lo:hi] @ self.tx.T).cpu().numpy()

    def top(self, sims):
        if sims.shape[1] <= self.k:
            raise ValueError('Need more nodes than graph k')
        part = np.argpartition(-sims, self.k, axis=1)[:, :self.k+1]
        vals = np.take_along_axis(sims, part, axis=1)
        ordered = np.sort(vals, axis=1)
        self.boundary_ties += int(np.sum(ordered[:,0] == ordered[:,1]))
        # Legacy np.argpartition top-k semantics are retained in the primary run.
        part = np.argpartition(-sims, self.k-1, axis=1)[:, :self.k]
        return np.take_along_axis(sims, part, axis=1), part

    def append(self, batch):
        oldn = len(self.x)
        b = np.ascontiguousarray(batch, dtype=np.float32)
        self.x = np.concatenate([self.x, b])
        self.global_ids = np.r_[self.global_ids, np.arange(self.next_id,self.next_id+len(b))]
        self.next_id += len(b)
        if self.tx is not None:
            import torch
            self.tx = torch.cat([self.tx, torch.as_tensor(b, device=self.device)])
        sims = self.dot_rows(oldn, len(self.x))
        sims[np.arange(len(b)), np.arange(oldn,len(self.x))] = -np.inf
        ns, ni = self.top(sims)
        # Reverse similarities update earlier nodes, but never their issued scores.
        both = np.concatenate([self.ts, sims[:,:oldn].T], axis=1)
        inds = np.concatenate([self.ti,np.broadcast_to(np.arange(oldn,len(self.x)),(oldn,len(b)))], axis=1)
        part = np.argpartition(-both,self.k-1,axis=1)[:,:self.k]
        self.ts = np.concatenate([np.take_along_axis(both,part,axis=1),ns])
        self.ti = np.concatenate([np.take_along_axis(inds,part,axis=1),ni])
        return normalized_graph(self.ts,self.ti,self.gamma), sims, oldn

    def matrix(self):
        return normalized_graph(self.ts,self.ti,self.gamma)


def solve(w, y, mode, previous=None, alpha=.9, iters=15, tol=1e-8, max_iter=1000):
    y = np.asarray(y,dtype=np.float64)
    rhs = (1-alpha)*y
    u = np.zeros_like(y)
    if mode == 'L0':
        if previous is None:
            u = y.copy()  # exact legacy first-batch initialization
        else:
            u[:len(previous)] = previous
    niter = iters if mode != 'L2' else max_iter
    start = time.perf_counter()
    for i in range(niter):
        u = alpha*(w@u)+rhs
        if mode == 'L2':
            residual = np.linalg.norm(u-alpha*(w@u)-rhs)/max(np.linalg.norm(rhs),1e-300)
            if residual <= tol:
                break
    residual = np.linalg.norm(u-alpha*(w@u)-rhs)/max(np.linalg.norm(rhs),1e-300)
    return u, {'iterations':i+1,'relative_residual':float(residual),
               'converged':bool(residual <= tol),'solve_seconds':time.perf_counter()-start,
               'l2_error_bound':float(np.linalg.norm(u-alpha*(w@u)-rhs)/(1-alpha))}


def class_max_solve(w, classes, ns, alpha=.9, tol=1e-8, device='cpu', block=64):
    """Classwise Zhou propagation in bounded column blocks; no class is omitted."""
    n, nc = w.shape[0], int(np.max(classes))+1
    out = np.zeros(n)
    worst, sweeps, start = 0., 0, time.perf_counter()
    if device != 'cpu':
        import torch
        tw = torch.sparse_csr_tensor(torch.tensor(w.indptr,device=device),torch.tensor(w.indices,device=device),
                                     torch.tensor(w.data.astype(np.float64),device=device),size=w.shape)
    for lo in range(0,nc,block):
        hi = min(lo+block,nc)
        y = np.zeros((n,hi-lo),np.float64)
        ix = np.flatnonzero((classes>=lo)&(classes<hi))
        y[ix,classes[ix]-lo] = 1-alpha
        if device == 'cpu':
            u=np.zeros_like(y)
            for it in range(1000):
                nxt=alpha*(w@u)+y
                res=np.linalg.norm(nxt-alpha*(w@nxt)-y,axis=0)/np.maximum(np.linalg.norm(y,axis=0),1e-300)
                u=nxt
                if res.max()<=tol:break
            out=np.maximum(out,u.max(1));r=float(res.max())
        else:
            rhs=torch.tensor(y,device=device);u=torch.zeros_like(rhs)
            den=torch.linalg.vector_norm(rhs,dim=0).clamp_min(1e-300)
            for it in range(1000):
                u=alpha*torch.sparse.mm(tw,u)+rhs
                if (it+1)%5==0:
                    res=torch.linalg.vector_norm(u-alpha*torch.sparse.mm(tw,u)-rhs,dim=0)/den
                    if float(res.max())<=tol:break
            r=float(res.max());out=np.maximum(out,u.max(1).values.cpu().numpy())
        worst=max(worst,r);sweeps=max(sweeps,it+1)
    return out,{'iterations':sweeps,'relative_residual':worst,'converged':worst<=tol,
                'solve_seconds':time.perf_counter()-start,'classes':nc}


class Memory:
    """m=1 entrance, all decisions use prior batches. Four disjoint roles optional."""
    def __init__(self,dc,dac,nc,med,mad,thresholds,split=False,capacity=None):
        self.dc,self.dac,self.nc=np.asarray(dc),np.asarray(dac),nc
        self.med,self.mad,self.thresholds=med,mad,thresholds
        self.roles=[np.arange(nc)]*4 if not split else [np.arange(j,nc,4) for j in range(4)]
        self.mem=[[] for _ in range(3)]
        self.best_cal=[np.full(nc,-np.inf,np.float32) for _ in range(3)]
        self.capacity=capacity

    def score(self,d,da,sims,ns,nfix,start):
        # sims = current batch x [support, calibration, past stream, current batch]
        ps=[p_high(self.dac[self.roles[0]],da)]
        for j in range(3):
            ids=self.mem[j]
            if ids:
                # Preserve legacy stream float32 subtraction, calibration float64 subtraction.
                rs=(1.-sims[:,nfix+np.asarray(ids)].max(1)).astype(np.float64)
            else:rs=np.full(len(d),2.)
            rc=np.where(np.isfinite(self.best_cal[j]),1.-self.best_cal[j].astype(np.float64),2.)
            g=d-(rs-self.med)/self.mad
            gc=self.dc-(rc-self.med)/self.mad
            ps.append(p_high(gc[self.roles[j+1]],g))
        masks=[ps[j]<=self.thresholds[j] for j in range(3)]
        sizes=[len(x) for x in self.mem]
        # Update after all scores and admissions for this batch have been issued.
        for j,mask in enumerate(masks):
            selected=np.flatnonzero(mask)
            if len(selected):
                self.mem[j].extend((start+selected).tolist())
                self.best_cal[j]=np.maximum(self.best_cal[j],sims[selected,ns:nfix].max(0))
        return ps,masks,sizes


def run_view(support,cal,sf,bidx,static,thresholds,device='cpu',classwise=False,
             modes=('L0','L1','L2'),k=10,alpha=.9,gamma=1.,snapshot=False):
    sup=np.asarray(support,np.float32).reshape(-1,support.shape[-1])
    cal=np.asarray(cal,np.float32);sf=np.asarray(sf,np.float32)
    ns,nc,nt=len(sup),len(cal),len(sf);nfix=ns+nc
    tick=time.perf_counter();graph=PrefixGraph(np.concatenate([sup,cal]),k,gamma,device)
    fixed_seconds=time.perf_counter()-tick
    y=np.r_[np.ones(ns),np.zeros(nc)]
    frozen={m:solve(graph.matrix(),y,m,alpha=alpha)[0][ns:] for m in modes}
    if classwise:
        cls=np.repeat(np.arange(len(support)),support.shape[1])
        frozen_class=class_max_solve(graph.matrix(),cls,ns,alpha,device=device)[0][ns:]
    mems={name:Memory(static['d_cal'],static['d_all_cal'],nc,static['stats']['med_all'],
                      static['stats']['mad_all'],thresholds,split=name=='M1') for name in ['M0','M1']}
    arr={name:np.empty(nt,np.float64) for name in ['static','pall','M0_pA1','M0_pA2','M0_pt','M1_pall','M1_pA1','M1_pA2','M1_pt']}
    arr['static'][:]=static['p'][nc:];arr['pall'][:]=static['p_all'][nc:]
    for mn in mems:
        for st in range(3):arr[f'{mn}_admit{st}']=np.empty(nt,bool)
    for m in modes:
        for typ in ['raw','norm','cdf','plp']:arr[f'{m}_{typ}']=np.empty(nt,np.float64)
    if classwise:
        for typ in ['raw','norm','cdf','plp']:arr[f'class_L2_{typ}']=np.empty(nt,np.float64)
    prev={m:None for m in modes};diag=[];cal_trace={m:[] for m in modes}
    for batch in np.unique(bidx):
        rows=np.flatnonzero(bidx==batch);start=int(rows[0]);assert np.array_equal(rows,np.arange(start,start+len(rows)))
        tick=time.perf_counter();w,sims,oldn=graph.append(sf[rows]);graph_seconds=time.perf_counter()-tick
        new=np.arange(oldn,w.shape[0]);yy=np.r_[np.ones(ns),np.zeros(w.shape[0]-ns)]
        mt=time.perf_counter()
        for mn,mem in mems.items():
            ps,masks,sizes=mem.score(static['d'][nc+rows],static['d_all'][nc+rows],sims,ns,nfix,start)
            for key,p in zip(['pall','pA1','pA2','pt'],ps):
                dest='pall' if mn=='M0' and key=='pall' else f'{mn}_{key}'
                arr[dest][rows]=p
            for j,mask in enumerate(masks):arr[f'{mn}_admit{j}'][rows]=mask
        memory_seconds=time.perf_counter()-mt
        for m in modes:
            u,dd=solve(w,yy,m,prev[m],alpha=alpha)
            prev[m]=u.copy()
            cu=u[ns:nfix];uu=u[new];den=max(float(np.median(u[:ns])),1e-12)
            arr[f'{m}_raw'][rows]=uu
            arr[f'{m}_norm'][rows]=uu/den
            arr[f'{m}_cdf'][rows]=p_low(frozen[m],uu)
            arr[f'{m}_plp'][rows]=p_low(cu,uu)
            cal_trace[m].append(cu.copy())
            so=np.argsort(uu,kind='stable');pp=arr[f'{m}_plp'][rows][so]
            dd.update(batch=int(batch),solver=m,n_nodes=w.shape[0],nnz=w.nnz,graph_seconds=graph_seconds,
                      fixed_graph_seconds=fixed_seconds if batch==bidx[0] else 0.,memory_seconds=memory_seconds,
                      support_median=den,cal_q01=float(np.quantile(cu,.01)),cal_q50=float(np.median(cu)),
                      raw_median=float(np.median(uu)),raw_tie_fraction=1-len(np.unique(uu))/len(uu),
                      p_tie_fraction=1-len(np.unique(pp))/len(pp),rank_reversals=int(np.sum(np.diff(pp)<0)),
                      boundary_ties=graph.boundary_ties)
            diag.append(dd)
        if classwise:
            u,dd=class_max_solve(w,cls,ns,alpha,device=device)
            for typ,val in [('raw',u[new]),('norm',u[new]/max(float(np.median(u[:ns])),1e-12)),
                            ('cdf',p_low(frozen_class,u[new])),('plp',p_low(u[ns:nfix],u[new]))]:arr[f'class_L2_{typ}'][rows]=val
            dd.update(batch=int(batch),solver='class_L2',n_nodes=w.shape[0],nnz=w.nnz,graph_seconds=graph_seconds)
            diag.append(dd)
        if snapshot:arr[f'snapshot_{int(batch)}']=prev[modes[0]].copy()
    return arr,diag,{m:np.array(v) for m,v in cal_trace.items()}
