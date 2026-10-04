import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from pilot import load_development
from core import run_view,p_high
from vins import r5
from vins.features import load_features
H=Path(__file__).resolve().parent;BASE=Path('/home/omote/vins_gonogo_20260925');CACHE=H/'data/real_calibration'


def load_identities(path):
    """Read our own hashed cache, including legacy pandas object-string arrays."""
    with np.load(path,allow_pickle=True) as cache:
        return {key:cache[key].astype(str) if key.endswith('_ids') else cache[key] for key in cache.files}


def prepare():
    if (CACHE/'ready.json').exists():raise FileExistsError(CACHE)
    CACHE.mkdir(parents=True,exist_ok=True);R,D=load_development()
    samples=pd.read_parquet(BASE/'splits/samples.parquet').set_index('sample_id')
    shots=pd.read_parquet(BASE/'r5/shots/draws.parquet')
    classes=json.loads((BASE/'splits/id_classes.json').read_text());order={c['idx_1k']:c['id_idx'] for c in classes}
    shots=shots[shots.idx_1k.isin(order)].assign(id_idx=lambda x:x.idx_1k.map(order))
    sup=shots[(shots.draw==0)&(shots.role=='support')].sort_values(['id_idx','pos'])
    pool=shots[~((shots.draw==0)&(shots.role=='support'))].sort_values(['id_idx','draw','pos'])
    assert np.all(pool.groupby('id_idx').size().values==68)
    rng=np.random.default_rng(20260927)
    selected=[]
    for cls in range(900):
        g=samples[(samples.split=='id_dev')&(samples.class_idx_id==cls)]
        selected += g.sample(5,random_state=int(rng.integers(1<<30))).index.tolist()
    oid=[]
    for _,g in samples[samples.split=='near_dev'].groupby('wnid'):
        oid += g.sample(15,random_state=int(rng.integers(1<<30))).index.tolist()
    ids=np.array(selected+oid);perm=rng.permutation(len(ids));ids=ids[perm]
    flag=np.r_[np.zeros(len(selected),bool),np.ones(len(oid),bool)][perm]
    cls=np.array([int(samples.loc[s].class_idx_id) if not o else -1 for s,o in zip(ids,flag)])
    clip,cb=load_features(BASE/'r5/features/shots.clip.pt');cp={s:i for i,s in enumerate(cb['sample_id'])}
    pos=torch.load(BASE/'runs/setup/tins_setup.pt',map_location='cpu')['positive_features'].float()
    pool_sim=(clip[[cp[s] for s in pool.sample_id]].float()@pos.T).numpy()
    srows=np.array([D['stream_row'][s] for s in ids]);cand=np.r_[np.argsort(-pool_sim,axis=1)[:,:20],np.argsort(-D['st_sims'][srows],axis=1)[:,:20]]
    np.savez(CACHE/'identities.npz',pool_ids=np.asarray(pool.sample_id.tolist(),dtype=str),support_ids=np.asarray(sup.sample_id.tolist(),dtype=str),
             stream_ids=ids.astype(str),is_ood=flag,id_class=cls,pool_class=pool.id_idx.values)
    for name,F in D['feats'].items():
        su=F['shots'][[F['shot_pos'][s] for s in sup.sample_id]].astype(np.float32).reshape(900,12,-1)
        pf=F['shots'][[F['shot_pos'][s] for s in pool.sample_id]].astype(np.float32)
        sf=F['stream'][srows];q=np.r_[pf,sf];is_cal=np.arange(len(q))<len(pf)
        v=r5.proto_view(su,q,cand,is_cal,n0=48,m=1)
        for key,value in [('support',su),('pool',pf),('stream',sf),('d',v['d']),('da',v['d_all'])]:np.save(CACHE/f'{name}_{key}.npy',value)
        (CACHE/f'{name}_stats.json').write_text(json.dumps(v['stats']))
    manifest={'n_support':10800,'n_pool':len(pool),'n_stream':len(ids),'n_id':int((~flag).sum()),'n_ood':int(flag.sum()),
              'pool_per_class':68,'source':'Previously used dev1; fixed support draw0; no new labels',
              'ids_sha256':hashlib.sha256((CACHE/'identities.npz').read_bytes()).hexdigest()}
    (CACHE/'ready.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def run(rep):
    torch.set_num_threads(1);start=time.perf_counter();cfg=json.loads((H/'preregistration.json').read_text())
    # This is our own hashed preparation artifact. pandas StringArray was saved
    # as object dtype by NumPy; read it explicitly and emit plain Unicode below.
    ids=load_identities(CACHE/'identities.npz');rng=np.random.default_rng([972750,rep])
    pick=np.concatenate([np.flatnonzero(ids['pool_class']==c)[rng.choice(68,4,replace=False)] for c in range(900)])
    raw={};rates=[];timings=[]
    for view in ['B14','L14']:
        su=np.load(CACHE/f'{view}_support.npy');pf=np.load(CACHE/f'{view}_pool.npy',mmap_mode='r')
        sf0=np.load(CACHE/f'{view}_stream.npy');dd=np.load(CACHE/f'{view}_d.npy');da=np.load(CACHE/f'{view}_da.npy')
        stats=json.loads((CACHE/f'{view}_stats.json').read_text());n_pool=len(pf);cal=pf[pick]
        for condition in ['ID_only','mixed','class_burst']:
            ix=np.arange(len(sf0))
            if condition=='ID_only':ix=ix[~ids['is_ood']]
            if condition=='class_burst':
                id_positions=np.flatnonzero(~ids['is_ood'])
                ix[id_positions]=id_positions[np.argsort(ids['id_class'][id_positions],kind='stable')]
            flag=ids['is_ood'][ix];classes=ids['id_class'][ix];bi=np.arange(len(ix))//256
            vd=np.r_[dd[pick],dd[n_pool+ix]];va=np.r_[da[pick],da[n_pool+ix]];nc=len(pick)
            v={'d':vd,'d_all':va,'p':p_high(vd[:nc],vd),'p_all':p_high(va[:nc],va),
               'd_cal':vd[:nc],'d_all_cal':va[:nc],'stats':stats}
            a,diag,_=run_view(su,cal,sf0[ix],bi,v,tuple(cfg['thresholds_exact']),device='cuda:0')
            for d in diag:d.update(view=view,condition=condition,rep=rep)
            timings+=diag
            for key in ['static','pall','M0_pA1','M0_pA2','M0_pt','M1_pall','M1_pA1','M1_pA2','M1_pt','L0_plp','L1_plp','L2_plp']:
                raw[f'{view}_{condition}_{key}']=a[key]
                for alpha in [.01,.05,.1]:
                    masks=[('all',~flag),('early',(~flag)&(bi<(bi.max()+1)/2)),('late',(~flag)&(bi>=(bi.max()+1)/2))]
                    masks += [(f'class{c}',(~flag)&(classes==c)) for c in range(900)]
                    for window,mask in masks:
                        if mask.any():rates.append(dict(rep=rep,view=view,condition=condition,component=key,alpha=alpha,window=window,
                                                        rate=float(np.mean(a[key][mask]<=alpha)),n=int(mask.sum())))
            for key in ['M0_admit2','M1_admit2']:
                raw[f'{view}_{condition}_{key}']=a[key]
                for cat,mask in [('ID',~flag),('OOD',flag)]:
                    rates.append(dict(rep=rep,view=view,condition=condition,component=key,alpha=cfg['thresholds_exact'][2],window=cat,
                                      rate=float(np.mean(a[key][mask])) if mask.any() else np.nan,n=int(mask.sum())))
    out=H/'results/real_calibration';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'rep{rep}.npz',**raw,calibration_ids=ids['pool_ids'][pick].astype(str))
    pd.DataFrame(rates).to_csv(out/f'rep{rep}.csv',index=False)
    pd.DataFrame(timings).to_csv(out/f'rep{rep}_solver.csv',index=False)
    (out/f'rep{rep}.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'gpu_hours_reserved_conservative':(time.perf_counter()-start)/3600,
                                              'rep':rep,'calibration_seed':[972750,rep]}))
    print(json.dumps({'rep':rep,'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--rep',type=int);a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.rep)
