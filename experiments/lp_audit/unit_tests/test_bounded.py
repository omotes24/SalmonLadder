import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import unittest,numpy as np
from test_core import fixture
from core import run_view
from bounded import bounded_view,memory_baselines,prune_graph
from core import PrefixGraph


class BoundedTests(unittest.TestCase):
    def test_deletion_update_equals_exact_rebuild(self):
        su,ca,sf,bi,v=fixture();x=np.r_[su.reshape(36,16),ca,sf]
        g=PrefixGraph(x);keep=np.r_[np.arange(48),np.arange(75,len(x))]
        g=prune_graph(g,keep);ref=PrefixGraph(x[keep])
        np.testing.assert_allclose(g.matrix().toarray(),ref.matrix().toarray(),atol=1e-7)

    def test_large_capacity_equals_full(self):
        su,ca,sf,bi,v=fixture()
        a,_,_=run_view(su,ca,sf,bi,v,(.3,.2,.102))
        b,d=bounded_view(su,ca,sf,bi,v,(.3,.2,.102),capacity=128,device='cpu',classwise=False)
        for k in b:np.testing.assert_allclose(a[k],b[k],atol=1e-7,err_msg=k)

    def test_capacity_applies_to_graph_and_all_memories(self):
        su,ca,sf,bi,v=fixture();a,d=bounded_view(su,ca,sf,bi,v,(1.,1.,1.),capacity=20,device='cpu',classwise=False)
        self.assertTrue(all(x['n_nodes']<=48+20 for x in d))
        self.assertTrue(all(x['max_A1_A2_M']<=20 for x in d))
        self.assertTrue(np.isfinite(a['L2_raw']).all())

    def test_capped_prefix(self):
        su,ca,sf,bi,v=fixture();a,_=bounded_view(su,ca,sf,bi,v,(.3,.2,.102),capacity=20,device='cpu',classwise=False)
        vv={k:(x[:44] if isinstance(x,np.ndarray) and x.shape[0]==76 else x) for k,x in v.items()}
        b,_=bounded_view(su,ca,sf[:32],bi[:32],vv,(.3,.2,.102),capacity=20,device='cpu',classwise=False)
        for k in b:np.testing.assert_array_equal(a[k][:32],b[k],err_msg=k)

    def test_memory_baselines_bounded_and_causal(self):
        su,ca,sf,bi,v=fixture();a,d=memory_baselines(su,ca,sf,bi,capacity=20,device='cpu')
        b,_=memory_baselines(su,ca,sf[:32],bi[:32],capacity=20,device='cpu')
        self.assertTrue(all(x['memory_size']<=20 for x in d))
        for k in b:np.testing.assert_array_equal(a[k][:32],b[k])


if __name__=='__main__':unittest.main(verbosity=2)
