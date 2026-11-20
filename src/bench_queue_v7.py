"""Compare time-conditioned onset with accepted v3 onset on disjoint months."""
import argparse,json
from pathlib import Path
import pandas as pd
from src.bench_queue_v2 import proxy_windows
from src.bench_queue import window_iou
from src.build_queue import train_onset_counts
from src.queue import predict_window
from src.queue_v7 import predict_onset

p=argparse.ArgumentParser();p.add_argument('--month',required=True);a=p.parse_args()
root=Path('data/release');rows=[]
settings=json.loads(Path('config/queue_settings.json').read_text())
overrides=json.loads(Path('config/queue_v2_overrides.json').read_text())
counts=json.loads(Path('reports/queue-event-counts-through-dec.json').read_text())
events=json.loads(Path('config/queue_v7_events.json').read_text())
assert set(events)==set(settings)
for panel,base in sorted(settings.items()):
    fd=pd.read_csv(root/'corridors'/panel/'network/fd_parameters.csv')
    cut=dict(zip(fd.link_id,.6*fd.free_speed_kmh))
    change=overrides.get(panel,{})
    if change.get('onset_source')=='events':
        c=counts[panel]['links'];ranking=sorted(c,key=lambda l:(-c[l],l))
    else:ranking=[l for l,_ in train_onset_counts(root,panel,cut).most_common()]
    ranking=ranking[:change.get('onset_top_k',base['onset_top_k'])]
    for condition,origin,history,future in proxy_windows(root,panel,a.month,cut,max_per_condition=31):
        if condition!='queue_onset':continue
        target=future[['timestamp','link_id']]
        candidates={'v6':predict_window(history,target,origin,condition,cut,ranking)}
        for bandwidth in [60,120,240]:
            candidates[f'time_{bandwidth}']=predict_onset(target,events[panel],origin,bandwidth)
        for model,pred in candidates.items():
            rows.append(dict(panel=panel,month=a.month,origin=origin,model=model,iou=window_iou(future,pred),n_pred=int(pred.queue_pred.sum()),n_true=int(future.queue_true.sum())))
    pd.DataFrame(rows).to_csv(f'reports/queue-v7-{a.month}.csv',index=False)
    print(panel,'done',flush=True)
print(pd.DataFrame(rows).groupby(['panel','model']).iou.mean().unstack().to_string(),flush=True)
