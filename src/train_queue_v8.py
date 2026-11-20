"""Fit per-corridor boosted queue classifiers and test chronological months."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from src.bench_queue_v2 import proxy_windows
from src.bench_queue import window_iou
from src.queue_v8 import features,baseline

root=Path('data/release');out=Path('outputs/v8/models');out.mkdir(parents=True,exist_ok=True)
settings=json.loads(Path('config/queue_settings.json').read_text())
v2=json.loads(Path('config/queue_v2_overrides.json').read_text());v3=json.loads(Path('config/queue_v3_overrides.json').read_text())
rows=[]
for panel in sorted(settings):
    fd=pd.read_csv(root/'corridors'/panel/'network/fd_parameters.csv')
    order=pd.read_csv(root/'corridors'/panel/'network/lwr_mainline_topology.csv').sort_values('order_index').mainline_link_id.astype(str).tolist()
    cut=dict(zip(fd.link_id.astype(str),.6*fd.free_speed_kmh))
    xs=[];ys=[]
    for month in [f'2030-{m:02d}' for m in range(6,13)]:
        for condition,origin,history,future in proxy_windows(root,panel,month,cut,max_per_condition=31):
            if condition!='queue_ongoing':continue
            valid=future.is_score_eligible.astype(bool).to_numpy()
            x=features(history,future[['timestamp','link_id']],origin,order,fd)
            xs.append(x[valid]);ys.append(future.queue_true.to_numpy(dtype=np.uint8)[valid])
    x=np.concatenate(xs);y=np.concatenate(ys)
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,max_depth=5,learning_rate=.08,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=81)
    model.fit(x,y);joblib.dump(model,out/f'{panel}.joblib')
    print(panel,'fit',len(y),'rows',int(y.sum()),'positive',flush=True)
    del xs,ys,x,y
    for month in ['2031-01','2031-02']:
        for condition,origin,history,future in proxy_windows(root,panel,month,cut,max_per_condition=31):
            if condition!='queue_ongoing':continue
            target=future[['timestamp','link_id']]
            base=baseline(history,target,origin,panel,cut,order,settings,v2,v3)
            rows.append(dict(panel=panel,month=month,origin=origin,model='v7',iou=window_iou(future,base)))
            probability=model.predict_proba(features(history,target,origin,order,fd))[:,1]
            for threshold in [.2,.3,.4,.5,.6]:
                pred=target.assign(queue_pred=(probability>=threshold).astype(int))
                rows.append(dict(panel=panel,month=month,origin=origin,model=f'ml_{threshold}',iou=window_iou(future,pred)))
    pd.DataFrame(rows).to_csv('reports/queue-v8-backtest.csv',index=False)
    print(pd.DataFrame(rows).query('panel==@panel').groupby(['month','model']).iou.mean().unstack().to_string(),flush=True)
