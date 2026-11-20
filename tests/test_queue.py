import unittest

import pandas as pd

from src.queue import predict_window


class QueueContractTests(unittest.TestCase):
    def setUp(self):
        self.origin = "2030-01-01T09:00:00Z"
        self.history = pd.DataFrame([
            ("2030-01-01T08:50:00Z", "A", 50., 1),
            ("2030-01-01T08:50:00Z", "B", 90., 1),
            ("2030-01-01T08:55:00Z", "A", 30., 1),
            ("2030-01-01T08:55:00Z", "B", 90., 1),
        ], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        self.targets = pd.DataFrame([
            ("2030-01-01T09:05:00Z", "A"),
            ("2030-01-01T09:05:00Z", "B"),
            ("2030-01-01T09:30:00Z", "A"),
            ("2030-01-01T09:30:00Z", "B"),
        ], columns=["timestamp", "link_id"])

    def test_ongoing_uses_last_visible_state(self):
        got = predict_window(self.history, self.targets, self.origin, "queue_ongoing",
                             {"A": 40., "B": 40.}, ["B"])
        self.assertEqual(got.queue_pred.tolist(), [1, 0, 1, 0])

    def test_onset_activates_only_final_horizon(self):
        got = predict_window(self.history, self.targets, self.origin, "queue_onset",
                             {"A": 40., "B": 40.}, ["B"])
        self.assertEqual(got.queue_pred.tolist(), [0, 0, 0, 1])

    def test_future_state_cannot_enter_history(self):
        leaked = pd.concat([self.history, pd.DataFrame([{"timestamp": self.origin,
            "link_id": "A", "speed_kmh": 0., "is_score_eligible": 1}])], ignore_index=True)
        with self.assertRaises(ValueError):
            predict_window(leaked, self.targets, self.origin, "queue_ongoing",
                           {"A": 40., "B": 40.}, ["B"])


if __name__ == "__main__":
    unittest.main()
