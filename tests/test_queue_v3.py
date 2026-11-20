import unittest

import pandas as pd

from src.queue_v3 import predict_ongoing_lookback


class QueueLookbackTests(unittest.TestCase):
    def test_carries_latest_eligible_link_state(self):
        history = pd.DataFrame([
            ("2030-01-01T08:50:00Z", "A", 30., 1),
            ("2030-01-01T08:50:00Z", "B", 80., 1),
            ("2030-01-01T08:55:00Z", "A", float("nan"), 0),
            ("2030-01-01T08:55:00Z", "B", 30., 1),
        ], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        targets = pd.DataFrame([("2030-01-01T09:05:00Z", "A"),
                                ("2030-01-01T09:05:00Z", "B")],
                               columns=["timestamp", "link_id"])
        got = predict_ongoing_lookback(history, targets, "2030-01-01T09:00:00Z",
                                         {"A": 40., "B": 40.}, lookback_steps=2)
        self.assertEqual(got.queue_pred.tolist(), [1, 1])

    def test_lookback_one_matches_last_frame_rule(self):
        history = pd.DataFrame([
            ("2030-01-01T08:50:00Z", "A", 30., 1),
            ("2030-01-01T08:55:00Z", "A", float("nan"), 0),
        ], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        targets = pd.DataFrame([("2030-01-01T09:05:00Z", "A")],
                               columns=["timestamp", "link_id"])
        got = predict_ongoing_lookback(history, targets, "2030-01-01T09:00:00Z",
                                         {"A": 40.}, lookback_steps=1)
        self.assertEqual(got.queue_pred.tolist(), [0])

    def test_downstream_extension_starts_at_fifth_step(self):
        history = pd.DataFrame([
            ("2030-01-01T08:55:00Z", "A", 80., 1),
            ("2030-01-01T08:55:00Z", "B", 30., 1),
            ("2030-01-01T08:55:00Z", "C", 80., 1),
        ], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        targets = pd.DataFrame([(f"2030-01-01T09:{minute:02d}:00Z", link)
                                for minute in (5, 10, 15, 20, 25, 30)
                                for link in ("A", "B", "C")],
                               columns=["timestamp", "link_id"])
        got = predict_ongoing_lookback(history, targets, "2030-01-01T09:00:00Z",
                                         {"A": 40., "B": 40., "C": 40.},
                                         lookback_steps=3, link_order=["A", "B", "C"],
                                         direction="downstream", radius=1, start_step=5)
        self.assertEqual(got.groupby("timestamp").queue_pred.sum().tolist(),
                         [1, 1, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
