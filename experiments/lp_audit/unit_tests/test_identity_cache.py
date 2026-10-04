import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from real_calibration import load_identities


class IdentityTests(unittest.TestCase):
    def test_legacy_object_ids_round_trip_without_pickle_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'old.npz';out=Path(tmp)/'output.npz'
            np.savez(src,pool_ids=np.array(['image/a.jpg','image/b.jpg'],object),pool_class=np.array([0,1]))
            ids=load_identities(src)
            np.testing.assert_array_equal(ids['pool_ids'],['image/a.jpg','image/b.jpg'])
            np.savez_compressed(out,calibration_ids=ids['pool_ids'][[1,0]])
            with np.load(out,allow_pickle=False) as checked:
                np.testing.assert_array_equal(checked['calibration_ids'],['image/b.jpg','image/a.jpg'])


if __name__=='__main__':unittest.main()
