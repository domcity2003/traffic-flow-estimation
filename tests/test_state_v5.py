import unittest
import numpy as np
from src.state_v5 import bounded_blend

class BoundedBlendTests(unittest.TestCase):
    def test_extreme_extrapolation_has_limited_influence(self):
        np.testing.assert_allclose(bounded_blend([80,80,80],[272,0,82],5),[82.5,77.5,81])

    def test_flow_limit_scales_by_lane_count(self):
        lane=np.array([2,4,8])
        base=np.full(3,1000.)
        got=bounded_blend(base,[2000,2000,2000],50*lane)
        np.testing.assert_allclose((got-base)/lane,[25,25,25])

    def test_nonnegative_inputs_remain_nonnegative(self):
        self.assertTrue((bounded_blend([0,1,100],[0,0,0],5)>=0).all())

if __name__=='__main__':unittest.main()
