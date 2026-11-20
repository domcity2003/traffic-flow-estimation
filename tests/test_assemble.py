import unittest

import pandas as pd

from src.assemble import fill_chunk


class AssemblyContractTests(unittest.TestCase):
    def setUp(self):
        self.key = pd.DataFrame([
            (1, "state", "P", "T", "S", "L", "R1", "-", "-", "-"),
            (2, "queue", "-", "T2", "-", "L", "-", "W", "-", "-"),
            (3, "odme", "P", "-", "-", "-", "-", "-", "D", "PATH"),
        ], columns=["submission_id", "task", "panel", "timestamp", "station_id",
                    "link_id", "mask_regime", "window_id", "departure_time", "path_id"])
        self.tables = {
            "state": pd.DataFrame([("P", "T", "S", "L", "R1", 80., 200.)],
                                  columns=["panel", "timestamp", "station_id", "link_id",
                                           "mask_regime", "speed_kmh", "flow_vph"]),
            "queue": pd.DataFrame([("W", "T2", "L", 1)],
                                  columns=["window_id", "timestamp", "link_id", "queue_pred"]),
            "odme": pd.DataFrame([("P", "D", "PATH", 50.)],
                                 columns=["panel", "departure_time", "path_id", "path_flow"]),
        }

    def test_exact_natural_keys_fill_all_tasks(self):
        got = fill_chunk(self.key, self.tables)
        self.assertEqual(got.speed_kmh.tolist(), [80., 0., 0.])
        self.assertEqual(got.queue_pred.tolist(), [0., 1., 0.])
        self.assertEqual(got.path_flow.tolist(), [0., 0., 50.])

    def test_missing_prediction_fails_instead_of_writing_zero(self):
        self.tables["queue"] = self.tables["queue"].iloc[0:0]
        with self.assertRaises(ValueError):
            fill_chunk(self.key, self.tables)


if __name__ == "__main__":
    unittest.main()
