"""Deterministic label-aware experiment construction, strictly separate from the detector."""
import json,time,itertools,hashlib
import numpy as np
from data import ROOT,Bank,dump

def write_plan(name,**arrays):
    p=ROOT/'plans'/f'{name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():np.savez_compressed(p,**arrays)
    return str(p)

def make_plans():
    if (ROOT/'plans/tasks.json').exists():return
    b=Bank('openood').load(['B14','L14']);f=b.frame
    idpool=np.flatnonzero((f.role=='eval')&f.eligible&~f.is_ood)
    rng=np.random.default_rng(2026092704)
    id_eval=rng.choice(idpool,128,replace=False);id_hist_pool=np.setdiff1d(idpool,id_eval)
    difficulty=np.log(np.maximum(b.views['B14']['p']*b.views['L14']['p'],1e-20))
    tasks=[];cohorts=[]
    for ds in ['ninco','ssb_hard']:
        op=np.flatnonzero((f.role=='eval')&f.eligible&(f.dataset==ds));cl=f['class'].to_numpy()
        possible=sorted(c for c in np.unique(cl[op]) if np.sum(cl[op]==c)>=40)
        if len(possible)<4:raise RuntimeError(f'{ds}: fewer than four eligible classes')
        for ci in rng.choice(possible,4,replace=False):
            own=op[cl[op]==ci];own=rng.permutation(own)
            queries=own[:8];recur=own[8:28];own_peers=own[28:33]
            other=op[cl[op]!=ci]
            for rep in [0,1]:
                rr=np.random.default_rng([20260927,4,len(cohorts),rep])
                avail=list(rr.permutation(other));donors=[]
                for x in recur:
                    at=int(np.argmin(abs(difficulty[avail]-difficulty[x])));donors.append(avail.pop(at))
                peer_donors=[]
                for x in own_peers:
                    at=int(np.argmin(abs(difficulty[avail]-difficulty[x])));peer_donors.append(avail.pop(at))
                avail=rr.permutation(avail);background=avail[:236];peers=np.r_[peer_donors,avail[236:486]]
                assert len(peers)==255
                history=np.r_[rr.choice(id_hist_pool,768,False),background,donors]
                history=history[rr.permutation(1024)]
                positions=np.array([np.flatnonzero(history==d)[0] for d in donors])
                cid=f'{ds}_c{len(cohorts):02d}_rep{rep}'
                base={'queries':queries,'id_eval':id_eval,'recur':recur,'donors':np.asarray(donors),'positions':positions,
                      'peers':peers,'own_peers':own_peers,'match_abs_logp_difference':abs(difficulty[recur]-difficulty[donors])}
                for r in [0,1,2,5,10,20]:
                    h=history.copy();h[positions[:r]]=recur[:r]
                    assert len(h)==1024 and len(np.unique(h))==1024 and f.is_ood.iloc[h].sum()==256
                    assert np.sum(cl[h]==ci)==r
                    assert not set(h)&set(np.r_[queries,id_eval,peers,own_peers])
                    pp=write_plan(f'recurrence_{cid}_r{r}',history=h,**base)
                    for batch,solver in itertools.product([1,16,256],['warm15','converged']):
                        tasks.append({'kind':'recurrence','id':f'R_{cid}_r{r}_b{batch}_{solver}',
                                      'bank':'openood','models':['B14','L14'],'plan':pp,'r':r,'batch':batch,'solver':solver,
                                      'dataset':ds,'target_class':str(ci),'rep':rep,'cap':None})
            cohorts.append({'dataset':ds,'class':str(ci),'query_ids':f.sample_id.iloc[queries].tolist()})
    # Two fixed unknown classes per panel. 40 OOD images at r=20 fit 0.1% in <45,000 distinct ID images.
    nin=np.flatnonzero((f.role=='eval')&f.eligible&(f.dataset=='ninco'))
    cl=f['class'].to_numpy();candidates=[c for c in sorted(np.unique(cl[nin])) if np.sum(cl[nin]==c)>=20]
    r5=np.random.default_rng(2026092705);selected=r5.choice(candidates,8,replace=False)
    runtime=[]
    for panel in range(4):
        cs=selected[2*panel:2*panel+2]
        per=[r5.permutation(nin[cl[nin]==c])[:20] for c in cs]
        ip=r5.permutation(idpool)
        for repeats,pattern in itertools.product([1,5,20],['random','burst','long_gap']):
            oo=np.concatenate([x[:repeats] for x in per]);n_ood=len(oo)
            order=np.random.default_rng([20260927,5,panel,repeats]).permutation(n_ood) if pattern=='random' else np.arange(n_ood)
            oo=oo[order]
            for fraction in [.001,.01,.05,.2]:
                n=int(round(n_ood/fraction));n_id=n-n_ood
                if n_id>len(ip):raise RuntimeError('insufficient unique ID images; never repeat images')
                rr=np.random.default_rng([20260927,50,panel,repeats])
                if pattern=='random':pos=np.sort(rr.choice(n,n_ood,replace=False))
                elif pattern=='burst':pos=np.r_[np.arange(int(.2*n),int(.2*n)+repeats),np.arange(int(.65*n),int(.65*n)+repeats)]
                else:
                    # Keep OOD image order fixed across fractions; only insert unique ID images.
                    # First half and last half of each class straddle a long gap.
                    half=max(1,repeats//2);pos=[];seq=[]
                    for phase,anchor in [(0,.05),(1,.8)]:
                        for c in range(2):
                            sel=per[c][:half] if phase==0 else per[c][half:repeats]
                            if phase==0:sel=per[c][:min(half,repeats)]
                            start=int((anchor+.08*c)*n);pos.extend(range(start,start+len(sel)));seq.extend(sel)
                    pos=np.array(pos);oo=np.array(seq)
                    sort=np.argsort(pos);pos=pos[sort];oo=oo[sort]
                assert len(pos)==n_ood and len(set(pos))==n_ood and pos.min()>=0 and pos.max()<n
                stream=np.empty(n,np.int64);stream[pos]=oo;mask=np.ones(n,bool);mask[pos]=False;stream[mask]=ip[:n_id]
                assert len(np.unique(stream))==len(stream)
                assert all(np.sum(cl[stream][f.is_ood.iloc[stream].values]==c)==repeats for c in cs)
                label=f'panel{panel}_r{repeats}_{pattern}_p{fraction:g}'
                pp=write_plan('lowfreq_'+label,stream=stream)
                for cap in [1000,5000,20000,None]:
                    task={'kind':'stream','id':f'L_{label}_w{cap}','bank':'openood','models':['B14','L14'],'plan':pp,
                          'batch':256,'solver':'warm15','cap':cap,'fraction':fraction,'repeats':repeats,'pattern':pattern,'panel':panel,
                          'long_gap_defined':repeats>1 if pattern=='long_gap' else True}
                    tasks.append(task)
                    if panel==0 and repeats==20 and (pattern=='random' or (pattern=='long_gap' and fraction==.001)):
                        rt=dict(task);rt.update(kind='runtime',id='E2E_'+task['id']);runtime.append(rt)
        pp=write_plan(f'zero_panel{panel}',stream=ip[:40000])
        for cap in [1000,5000,20000,None]:
            task={'kind':'stream','id':f'Z_panel{panel}_w{cap}','bank':'openood','models':['B14','L14'],'plan':pp,'batch':256,
                  'solver':'warm15','cap':cap,'fraction':0.,'repeats':0,'pattern':'ID_only','panel':panel}
            tasks.append(task)
            if panel==0:
                rt=dict(task);rt.update(kind='runtime',id='E2E_'+task['id']);runtime.append(rt)
    tasks+=runtime
    for name in ['cub','cifar']:
        for models in [['B14'],['CLIP'],['B14','L14']]:
            for seed in [123,124,125]:
                tasks.append({'kind':'holdout','id':f'H_{name}_{"-".join(models)}_s{seed}','bank':name,'models':models,'seed':seed,
                              'batch':256,'cap':None,'solver':'warm15'})
    # Favor a complete diagnostic curve early; low-frequency and holdout jobs then interleave.
    tasks.sort(key=lambda t: (0 if t['kind']=='recurrence' and t['batch']==256 else
                             1 if t['kind']=='holdout' else 2 if t['kind']=='stream' else 3 if t['kind']=='runtime' else 4,t['id']))
    dump(ROOT/'plans/cohorts.json',cohorts)
    dump(ROOT/'plans/tasks.json',tasks)
    dump(ROOT/'plans/design_receipt.json',{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'counts':{k:sum(t['kind']==k for t in tasks) for k in ['recurrence','stream','runtime','holdout']},
                'unique_id_pool':len(idpool),'class_panels':selected.tolist(),'r': [0,1,2,5,10,20],
                'same_class_difficulty_matching':'greedy nearest absolute difference of log(product static DINO p); differences saved',
                'score_labels_used_only_in_plan_and_metrics':True})

if __name__=='__main__':make_plans()
