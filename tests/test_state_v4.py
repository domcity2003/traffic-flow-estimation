import unittest
import numpy as np
from src.state_v4 import complete_grid

class CompletionTests(unittest.TestCase):
    def test_preserves_observed_and_recovers_shared_sensor_pattern(self):
        truth = np.outer(2+np.sin(np.arange(48)/8), [10.,20.,30.,40.,50.])
        masked=truth.copy()
        masked[10:20,2]=np.nan
        pred=complete_grid(masked,rank=1)
        np.testing.assert_array_equal(pred[np.isfinite(masked)],truth[np.isfinite(masked)])
        self.assertLess(np.mean(abs(pred[10:20,2]-truth[10:20,2])),.01)

    def test_entire_missing_sensor_and_time_stay_finite(self):
        grid=np.outer(np.arange(1.,11.),[1.,2.,3.,4.])
        grid[:,1]=np.nan
        grid[4,:]=np.nan
        pred=complete_grid(grid,rank=2)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred>=0).all())


class PredictionTests(unittest.TestCase):
    def test_wrapper_matches_manual_blend_and_target_order(self):
        import pandas as pd
        from src.state import predict_day as baseline
        from src.state_v4 import predict_day
        times=[f'2031-01-01T00:{i*5:02d}:00Z' for i in range(8)]
        grid=np.outer(np.arange(1.,9.),[10.,20.,30.])
        grid[3,1]=np.nan
        rows=[(time,link,link,grid[i,j],grid[i,j]*10,100) for i,time in enumerate(times) for j,link in enumerate(['A','B','C'])]
        masked=pd.DataFrame(rows,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph','pct_observed'])
        targets=pd.DataFrame([(times[3],'B','B','R1')],columns=['timestamp','station_id','link_id','mask_regime'])
        rules={'speed_kmh':{'rank':1,'blend':.5}}
        got=predict_day(masked,targets,['A','B','C'],rules=rules)
        base=baseline(masked,targets,['A','B','C'])
        expected=.5*base.speed_kmh.iloc[0]+.5*complete_grid(grid,rank=1)[3,1]
        self.assertAlmostEqual(got.speed_kmh.iloc[0],expected)
        self.assertEqual(got.flow_vph.iloc[0],base.flow_vph.iloc[0])
        self.assertEqual(got.link_id.tolist(),['B'])

if __name__=='__main__': unittest.main()
