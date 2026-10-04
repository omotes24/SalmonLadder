import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest,numpy as np
from metrics import metrics,log_nonnegative


class MetricTests(unittest.TestCase):
    def test_zero_products_keep_exact_ties(self):
        a=np.array([0.,0.,.2,.9]);b=np.array([.1,.8,.5,.9]);flag=np.array([1,1,0,0],bool)
        lp=log_nonnegative(a)+log_nonnegative(b)
        self.assertEqual(lp[0],lp[1]);self.assertEqual(metrics(lp,flag)['AUROC'],metrics(a*b,flag)['AUROC'])

    def test_ties_accept_at_least_target_id(self):
        score=np.r_[np.ones(20),np.zeros(5),np.ones(5)]
        flag=np.r_[np.zeros(20,bool),np.ones(10,bool)]
        m=metrics(score,flag);self.assertEqual(m['actual_ID_acceptance'],100);self.assertEqual(m['FPR95'],50)


if __name__=='__main__':unittest.main(verbosity=2)
