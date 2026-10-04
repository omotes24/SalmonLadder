import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'vendor')]
import unittest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from core import PrefixGraph, normalized_graph, solve, class_max_solve, p_low, p_high, run_view
from vins import r5


def fixture(seed=20):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(3*12+3*4+64,16)).astype(np.float32)
    a/=np.linalg.norm(a,axis=1,keepdims=True)
    sup=a[:36].reshape(3,12,16);cal=a[36:48];sf=a[48:]
    q=np.r_[cal,sf];cand=np.tile(np.arange(3),(len(q),1));mask=np.arange(len(q))<len(cal)
    v=r5.proto_view(sup,q,cand,mask,n0=48,m=1)
    return sup,cal,sf,np.arange(len(sf))//16,v


class CoreTests(unittest.TestCase):
    def test_legacy_equivalence(self):
        su,ca,sf,bi,v=fixture()
        a,d,_=run_view(su,ca,sf,bi,v,(.3,.2,.10191613435745239))
        ref=r5.lp_run(su,ca,sf,bi,k=10,alpha=.9,gamma=1.,iters=15)
        np.testing.assert_allclose(a['L0_raw'],ref['u'],atol=2e-8,rtol=2e-7)
        np.testing.assert_array_equal(a['L0_plp'],ref['p'])
        masks=r5.entrance(sf,v['d'][12:],v['p_all'][12:],bi,ca,v['d_cal'],
                         v['stats']['med_all'],v['stats']['mad_all'],(.3,.2,.10191613435745239),1)
        for i,m in enumerate(masks):np.testing.assert_array_equal(a[f'M0_admit{i}'],m)
        pt,_=r5.memory_p(sf,v['d'][12:],bi,masks[-1],ca,v['d_cal'],v['stats']['med_all'],v['stats']['mad_all'],1)
        np.testing.assert_array_equal(a['M0_pt'],pt)

    def test_prefix_and_immutable_scores(self):
        su,ca,sf,bi,v=fixture()
        a,_,_=run_view(su,ca,sf,bi,v,(.3,.2,.102))
        vs={k:(x[:44] if isinstance(x,np.ndarray) and x.shape[0]==76 else x) for k,x in v.items()}
        b,_,_=run_view(su,ca,sf[:32],bi[:32],vs,(.3,.2,.102))
        for k in b:np.testing.assert_array_equal(a[k][:32],b[k],err_msg=k)
        sf2=sf.copy();sf2[32:]=np.roll(sf2[32:],3,axis=1)
        q=np.r_[ca,sf2];v2=r5.proto_view(su,q,np.tile(np.arange(3),(len(q),1)),np.arange(len(q))<12,n0=48,m=1)
        c,_,_=run_view(su,ca,sf2,bi,v2,(.3,.2,.102))
        for k in a:np.testing.assert_array_equal(a[k][:32],c[k][:32],err_msg=k)

    def test_graph_prefix_against_rebuild(self):
        su,ca,sf,_,_=fixture();g=PrefixGraph(np.r_[su.reshape(-1,16),ca])
        for batch in [sf[:16],sf[16:32]]:
            w,_,_=g.append(batch);r=PrefixGraph(g.x)
            np.testing.assert_allclose(w.toarray(),r.matrix().toarray(),atol=1e-7)

    def test_linear_solution_and_error_bound(self):
        su,ca,sf,_,_=fixture();g=PrefixGraph(np.r_[su.reshape(-1,16),ca,sf]);w=g.matrix();y=np.r_[np.ones(36),np.zeros(76)]
        exact=spsolve(sp.eye(len(y))- .9*w.astype(np.float64),(1-.9)*y)
        for mode in ['L0','L1','L2']:
            u,d=solve(w,y,mode)
            self.assertLessEqual(np.linalg.norm(u-exact),d['l2_error_bound']*1.001+1e-12)
        self.assertTrue(d['converged']);np.testing.assert_allclose(u,exact,atol=1e-8)

    def test_permutation_equivariance_without_ties(self):
        su,ca,sf,_,_=fixture();x=np.r_[su.reshape(-1,16),ca,sf];y=np.r_[np.ones(36),np.zeros(76)]
        p=np.random.default_rng(12).permutation(len(x));ip=np.argsort(p)
        w=PrefixGraph(x).matrix();wp=PrefixGraph(x[p]).matrix()
        for m in ['L1','L2']:
            u,_=solve(w,y,m);up,_=solve(wp,y[p],m)
            np.testing.assert_allclose(u,up[ip],atol=1e-7)

    def test_split_calibration_no_feedback(self):
        su,ca,sf,bi,v=fixture();a,_,_=run_view(su,ca,sf,bi,v,(.3,.2,.102))
        vv={**v,'d_cal':v['d_cal'].copy(),'d_all_cal':v['d_all_cal'].copy()}
        vv['d_cal'][3::4]+=10;vv['d_all_cal'][3::4]+=10
        b,_,_=run_view(su,ca,sf,bi,vv,(.3,.2,.102))
        for j in range(3):np.testing.assert_array_equal(a[f'M1_admit{j}'],b[f'M1_admit{j}'])
        vv={**v,'d_cal':v['d_cal'].copy(),'d_all_cal':v['d_all_cal'].copy()};vv['d_cal'][2::4]+=10
        b,_,_=run_view(su,ca,sf,bi,vv,(.3,.2,.102))
        for j in [0,1]:np.testing.assert_array_equal(a[f'M1_admit{j}'],b[f'M1_admit{j}'])

    def test_monotone_rank_and_ties(self):
        p=p_low(np.array([.1,.2,.3]),np.array([0,.05,.1,.15,.2,.8]))
        self.assertTrue((np.diff(p)>=0).all());self.assertEqual(p[0],p[1])
        np.testing.assert_array_equal(p_high(np.array([.1,.2,.3]),[.1]),[1.])

    def test_classwise_all_classes_and_dense_reference(self):
        su,ca,sf,_,_=fixture();x=np.r_[su.reshape(-1,16),ca,sf];w=PrefixGraph(x).matrix();cls=np.repeat(np.arange(3),12)
        u,d=class_max_solve(w,cls,36,block=2)
        y=np.zeros((len(x),3));y[np.arange(36),cls]=.1
        exact=spsolve(sp.eye(len(x))-.9*w.astype(np.float64),y).max(1)
        np.testing.assert_allclose(u,exact,atol=1e-8);self.assertEqual(d['classes'],3)


if __name__=='__main__':unittest.main(verbosity=2)
