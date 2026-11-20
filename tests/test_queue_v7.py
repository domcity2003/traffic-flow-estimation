import unittest
import pandas as pd
from src.queue_v7 import choose_links,predict_onset

class OnsetTests(unittest.TestCase):
    def events(self):
        return [dict(origin=f'2030-12-{day:02d}T{hour}:00:00Z',queued=[link],eligible=['A','B']) for day in range(1,10) for hour,link in [('08','A'),('17','B')]]

    def test_time_selects_distinct_bottlenecks(self):
        self.assertEqual(choose_links(self.events(),'2031-01-01T08:00:00Z',60),['A'])
        self.assertEqual(choose_links(self.events(),'2031-01-01T17:00:00Z',60),['B'])

    def test_only_last_horizon_is_marked(self):
        targets=pd.DataFrame([(f'2031-01-01T08:{minute:02d}:00Z',link) for minute in [5,10,15,20,25,30] for link in ['A','B']],columns=['timestamp','link_id'])
        pred=predict_onset(targets,self.events(),'2031-01-01T08:00:00Z',60)
        self.assertEqual(pred.queue_pred.tolist(),[0]*10+[1,0])

    def test_future_training_event_rejected(self):
        with self.assertRaises(ValueError): choose_links(self.events(),'2030-01-01T08:00:00Z')

if __name__=='__main__':unittest.main()
