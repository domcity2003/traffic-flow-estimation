"""Apply frozen ongoing classifiers to v7 queue predictions."""
import json
from pathlib import Path
import joblib
import pandas as pd
from src.queue_v8 import features

root=Path('data/release')
settings=json.loads(Path('config/queue_v8_settings.json').read_text())
base=pd.read_csv('outputs/v7/queue.csv',dtype={'window_id':str,'timestamp':str,'link_id':str})
seen=set();changed=0
for panel,threshold in settings.items():
    fd=pd.read_csv(root/'corridors'/panel/'network/fd_parameters.csv')
    order=pd.read_csv(root/'corridors'/panel/'network/lwr_mainline_topology.csv').sort_values('order_index').mainline_link_id.astype(str).tolist()
    model=joblib.load(Path('outputs/v8/models')/f'{panel}.joblib')
    for split in ['validation','private']:
        p=root/'task2'/panel/split
        windows=pd.read_csv(p/'window_index.csv',dtype=str)
        history=pd.read_parquet(p/'window_history.parquet')
        histories={key:group for key,group in history.groupby('window_id')}
        for w in windows.itertuples(index=False):
            if w.condition!='queue_ongoing':continue
            mask=base.window_id==w.window_id
            if not mask.any() or w.window_id in seen:raise ValueError('missing or duplicate window')
            seen.add(w.window_id)
            target=base.loc[mask,['window_id','timestamp','link_id']]
            probability=model.predict_proba(features(histories[w.window_id],target,w.forecast_origin,order,fd))[:,1]
            pred=(probability>=threshold).astype(int)
            changed+=int((base.loc[mask,'queue_pred'].to_numpy()!=pred).sum())
            base.loc[mask,'queue_pred']=pred
        print(panel,split,'done',flush=True)
output=Path('outputs/v8/queue.csv');output.parent.mkdir(parents=True,exist_ok=True)
base.to_csv(output,index=False)
print('Rows',len(base),'changed',changed,'ongoing windows',len(seen),flush=True)
