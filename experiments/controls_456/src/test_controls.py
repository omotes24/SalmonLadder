"""Regression tests for leakage, native equivalence, convergence and eviction semantics."""
import sys,importlib.util
from pathlib import Path
import numpy as np
import torch
from core import Detector,Graph,static_view,norm,score_methods
from data import VENDOR
sys.path.insert(0,str(VENDOR))

def fixture():
    r=np.random.default_rng(705)
    f=norm(r.normal(size=(72,24))+.3)
    s=f[:36].reshape(3,12,24);cal=f[36:48];q=f[48:]
    cand=np.tile(np.arange(3),(36,1))
    v=static_view(s,cal,np.r_[cal,q],cand[:12],cand)
    return s,cal,q,v

def test_frozen_cpu_equivalence():
    s,cal,q,v=fixture()
    spec=importlib.util.spec_from_file_location('ev',VENDOR/'scripts/iter3_eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    cands=np.tile(np.arange(3),(len(cal)+len(q),1));ic=np.r_[np.ones(len(cal),bool),np.zeros(len(q),bool)]
    ref=ev.custom_view(s,np.r_[cal,q],cands,ic,np.repeat(np.arange(3),4),proto=True);ref['support_arr']=s
    assert np.allclose(v['d'],ref['d'],atol=1e-12)
    bidx=np.arange(len(q))//4
    am=ev.admit_mask(ref,q,ref['d'][12:],ref['p_all'][12:],bidx,'cand40_30_10')
    pt=ev.p_memory(ref,q,ref['d'][12:],bidx,am)[0]
    lp=ev.lp_pvalues(ref,q,bidx)
    det=Detector(v);ours=[];pls=[];m=[]
    for lo in range(0,len(q),4):
        rr=det.step(q[lo:lo+4],v['d'][12+lo:12+lo+4],v['pall'][12+lo:12+lo+4]);ours.extend(rr['memory']);pls.extend(rr['lp']);m.extend(rr['admit_M'])
    assert np.array_equal(m,am)
    assert np.array_equal(ours,pt)
    assert np.array_equal(pls,lp)

def test_probe_state_isolation():
    _,_,q,v=fixture();d=Detector(v)
    d.step(q[:8],v['d'][12:20],v['pall'][12:20]);before=d.graph.u.copy();sizes=[len(x) for x in d.banks]
    a=d.clone().step(q[8:9],v['d'][20:21],v['pall'][20:21])
    d.clone().step(q[10:11],v['d'][22:23],v['pall'][22:23])
    b=d.clone().step(q[8:9],v['d'][20:21],v['pall'][20:21])
    assert np.array_equal(a['full'],b['full']);assert np.array_equal(d.graph.u,before)
    assert sizes==[len(x) for x in d.banks] and d.seen==8

def test_all_retention_bounds_and_exact_neighbors():
    _,_,q,v=fixture();d=Detector(v,cap=7)
    for lo in range(0,len(q),3):
        d.step(q[lo:lo+3],v['d'][12+lo:12+lo+3],np.zeros(min(3,len(q)-lo)))
        assert all(len(x)<=7 for x in d.banks)
        assert all((a>=d.seen-7).all() for a in d.ages)
        g=d.graph;assert len(g.X)<=g.nfix+7
        val,ind=g._neighbors(g.X,torch.arange(len(g.X)))
        assert torch.allclose(g.s,val,atol=2e-6)
        assert torch.equal(g.i,ind)

def test_convergence_removes_batch_iteration_count_effect():
    s,c,q,_=fixture();states=[]
    for batch in [1,4,24]:
        g=Graph(s.reshape(-1,24),c,mode='converged')
        for lo in range(0,len(q),batch):g.append(q[lo:lo+batch])
        states.append(g.u)
        assert g.last['relative_error_bound']<=1e-6
    assert np.allclose(states[0],states[1],atol=1e-6)
    assert np.allclose(states[0],states[2],atol=1e-6)

def test_no_future_images_influence_past():
    _,_,q,v=fixture();a=Detector(v);b=Detector(v)
    x=a.step(q[:4],v['d'][12:16],v['pall'][12:16])
    y=b.step(q[:4],v['d'][12:16],v['pall'][12:16])
    a.step(q[4:],v['d'][16:],v['pall'][16:])
    assert np.array_equal(x['full'],y['full'])

def test_image_only_condition_ignores_text_candidates(tmp_path,monkeypatch):
    import data
    import pandas as pd
    base=tmp_path/'data'/'mini';base.mkdir(parents=True)
    rng=np.random.default_rng(91);features=norm(rng.normal(size=(120,32)))
    np.save(base/'CLIP.npy',features)
    pd.DataFrame({'sample_id':list(map(str,range(120)))}).to_parquet(base/'images.parquet')
    data.dump(base/'ready.json',{'ns':72,'nc':24,'nclass':6})
    monkeypatch.setattr(data,'ROOT',tmp_path)
    np.save(base/'cand.npy',np.zeros((120,5),np.int64))
    first=data.Bank('mini').load(['CLIP']).views['CLIP']['d'].copy()
    (base/'view_CLIP_imageonly.npz').unlink()
    np.save(base/'cand.npy',np.full((120,5),5,np.int64))
    second=data.Bank('mini').load(['CLIP']).views['CLIP']['d']
    assert np.array_equal(first,second)
