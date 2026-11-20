"""Patch accepted queue file with frozen time-conditioned onset predictions."""
import json
from pathlib import Path
import pandas as pd
from src.queue_v7 import predict_onset

root=Path('data/release')
settings=json.loads(Path('config/queue_v7_settings.json').read_text())
events=json.loads(Path('config/queue_v7_events.json').read_text())
base=pd.read_csv('outputs/v3/queue.csv',dtype={'window_id':str,'timestamp':str,'link_id':str})
seen=set();changed=0
for panel,bandwidth in settings.items():
    for split in ['validation','private']:
        windows=pd.read_csv(root/'task2'/panel/split/'window_index.csv',dtype=str)
        for w in windows.itertuples(index=False):
            if w.condition!='queue_onset':continue
            mask=base.window_id==w.window_id
            if not mask.any() or w.window_id in seen:raise ValueError('missing or duplicate window')
            seen.add(w.window_id)
            target=base.loc[mask,['window_id','timestamp','link_id']]
            pred=predict_onset(target,events[panel],w.forecast_origin,bandwidth)
            changed+=int((base.loc[mask,'queue_pred'].to_numpy()!=pred.queue_pred.to_numpy()).sum())
            base.loc[mask,'queue_pred']=pred.queue_pred.to_numpy()
        print(panel,split,'done',flush=True)
output=Path('outputs/v7/queue.csv');output.parent.mkdir(parents=True,exist_ok=True)
base.to_csv(output,index=False)
print('Rows',len(base),'changed',changed,'onset windows',len(seen),flush=True)
