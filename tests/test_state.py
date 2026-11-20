import unittest

import numpy as np
import pandas as pd

from src.state import interpolate_grid, predict_day


class StateInterpolationTests(unittest.TestCase):
    def test_temporal_interpolation_uses_visible_endpoints(self):
        grid = np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0]])
        got = interpolate_grid(grid, axis=0)
        np.testing.assert_allclose(got[1], [20.0, 30.0])

    def test_predict_day_emits_only_template_target(self):
        timestamps = ["2030-01-01T00:00:00Z", "2030-01-01T00:05:00Z", "2030-01-01T00:10:00Z"]
        rows = []
        for i, ts in enumerate(timestamps):
            rows.append((ts, "station-a", "A", [10.0, np.nan, 30.0][i], [100.0, np.nan, 300.0][i], 100))
            rows.append((ts, "station-b", "B", [20.0, 25.0, 30.0][i], [200.0, 250.0, 300.0][i], 100))
        masked = pd.DataFrame(rows, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        targets = pd.DataFrame([("2030-01-01T00:05:00Z", "station-a", "A", "R1")],
                               columns=["timestamp", "station_id", "link_id", "mask_regime"])
        got = predict_day(masked, targets, ["A", "B"], temporal_weight=0.8)
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["link_id"], "A")
        self.assertAlmostEqual(got.iloc[0]["speed_kmh"], 21.0)
        self.assertAlmostEqual(got.iloc[0]["flow_vph"], 210.0)

    def test_spatial_flow_respects_target_lane_count(self):
        masked = pd.DataFrame([
            ("2030-01-01T00:00:00Z", "A", 200.0, 100),
            ("2030-01-01T00:00:00Z", "B", 400.0, 100),
            ("2030-01-01T00:05:00Z", "A", np.nan, 0),
            ("2030-01-01T00:05:00Z", "B", 400.0, 100),
            ("2030-01-01T00:10:00Z", "A", 200.0, 100),
            ("2030-01-01T00:10:00Z", "B", 400.0, 100),
        ], columns=["timestamp", "link_id", "flow_vph", "pct_observed"])
        masked["station_id"] = masked.link_id
        masked["speed_kmh"] = 100.0
        target = pd.DataFrame([("2030-01-01T00:05:00Z", "A", "A", "R1")],
                              columns=["timestamp", "station_id", "link_id", "mask_regime"])
        got = predict_day(masked, target, ["A", "B"], temporal_weight=.5,
                          spatial_scale={"flow_vph": {"A": 2, "B": 4}})
        self.assertAlmostEqual(got.flow_vph.iloc[0], 200.0)

    def test_speed_and_flow_can_use_different_blends(self):
        timestamps = ["2030-01-01T00:00:00Z", "2030-01-01T00:05:00Z", "2030-01-01T00:10:00Z"]
        rows = []
        for i, ts in enumerate(timestamps):
            rows.extend([(ts, "A", [10., np.nan, 30.][i], [100., np.nan, 300.][i], 100),
                         (ts, "B", [20., 25., 30.][i], [200., 250., 300.][i], 100)])
        masked = pd.DataFrame(rows, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        masked["station_id"] = masked.link_id
        target = pd.DataFrame([(timestamps[1], "A", "A", "R1")],
                              columns=["timestamp", "station_id", "link_id", "mask_regime"])
        got = predict_day(masked, target, ["A", "B"], temporal_weight={"speed_kmh": 1., "flow_vph": 0.})
        self.assertAlmostEqual(got.speed_kmh.iloc[0], 20.)
        self.assertAlmostEqual(got.flow_vph.iloc[0], 250.)


if __name__ == "__main__":
    unittest.main()
