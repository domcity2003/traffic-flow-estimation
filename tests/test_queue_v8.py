import unittest
import numpy as np
import pandas as pd
from src.queue_v8 import features

class FeatureTests(unittest.TestCase):
    def sample(self):
        times=pd.date_range('2030-01-01T00:00:00Z',periods=12,freq='5min').astype(str)
        h=pd.DataFrame([(t,l,60+j,1000.,.2,True) for t in times for j,l in enumerate(['A','B','C'])],columns=['timestamp','link_id','speed_kmh','flow_vph','occupancy','is_score_eligible'])
        t=pd.DataFrame([('2030-01-01T01:05:00Z','C'),('2030-01-01T01:10:00Z','A')],columns=['timestamp','link_id'])
        fd=pd.DataFrame({'link_id':['A','B','C'],'free_speed_kmh':[100]*3,'capacity_vph':[2000]*3})
        return h,t,fd

    def test_order_and_horizon_preserved(self):
        h,t,fd=self.sample();x=features(h,t,'2030-01-01T01:00:00Z',['A','B','C'],fd)
        np.testing.assert_allclose(x[:,0],[.62,.6]);np.testing.assert_allclose(x[:,-3],[5,10])

    def test_future_history_rejected(self):
        h,t,fd=self.sample();h.loc[0,'timestamp']='2030-01-01T01:00:00Z'
        with self.assertRaises(ValueError):features(h,t,'2030-01-01T01:00:00Z',['A','B','C'],fd)

    def test_unknown_link_rejected(self):
        h,t,fd=self.sample();t.loc[0,'link_id']='unknown'
        with self.assertRaises(ValueError):features(h,t,'2030-01-01T01:00:00Z',['A','B','C'],fd)

if __name__=='__main__':unittest.main()
