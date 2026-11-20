import unittest

import pandas as pd

from src.queue_v2 import predict_ongoing_propagation


class QueuePropagationTests(unittest.TestCase):
    def test_upstream_expansion_only_after_selected_horizon(self):
        history = pd.DataFrame([
            ("2030-01-01T08:55:00Z", "A", 90., 1),
            ("2030-01-01T08:55:00Z", "B", 30., 1),
            ("2030-01-01T08:55:00Z", "C", 90., 1),
        ], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        targets = pd.DataFrame([(ts, link) for ts in
                                ("2030-01-01T09:05:00Z", "2030-01-01T09:10:00Z")
                                for link in ("A", "B", "C")], columns=["timestamp", "link_id"])
        got = predict_ongoing_propagation(history, targets, "2030-01-01T09:00:00Z",
                                           {link: 40. for link in ("A", "B", "C")},
                                           ["A", "B", "C"], direction="upstream",
                                           radius=1, start_step=2)
        self.assertEqual(got.queue_pred.tolist(), [0, 1, 0, 1, 1, 0])

    def test_rejects_observation_at_origin(self):
        history = pd.DataFrame([{"timestamp": "2030-01-01T09:00:00Z", "link_id": "A",
                                 "speed_kmh": 30., "is_score_eligible": 1}])
        targets = pd.DataFrame([{"timestamp": "2030-01-01T09:05:00Z", "link_id": "A"}])
        with self.assertRaises(ValueError):
            predict_ongoing_propagation(history, targets, "2030-01-01T09:00:00Z",
                                           {"A": 40.}, ["A"], direction="upstream",
                                           radius=1, start_step=1)


if __name__ == "__main__":
    unittest.main()
