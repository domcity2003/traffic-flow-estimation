"""Freeze thresholds and corridor adoption using January checks only."""
import json
from pathlib import Path
import pandas as pd
x=pd.read_csv('reports/queue-v8-backtest.csv')
s=x[x.month=='2031-01'].groupby(['panel','model']).iou.mean().unstack()
expected=set(json.loads(Path('config/queue_settings.json').read_text()))
if set(s.index)!=expected:raise ValueError('training/evaluation not complete')
extended=pd.read_csv('reports/queue-v8-extended.csv')
late=extended[extended.month=='2031-01'].groupby('panel')[['v7','v8']].mean()
if set(late.index)!=expected:raise ValueError('extended evaluation not complete')
choices={}
for panel,row in s.iterrows():
    model=row[[c for c in row.index if c.startswith('ml_')]].idxmax()
    if row[model]>row.v7+.025 and late.loc[panel,'v8']>late.loc[panel,'v7']:
        choices[panel]=float(model.split('_')[1])
Path('config/queue_v8_settings.json').write_text(json.dumps(choices,indent=2)+'\n')
for month,g in x.groupby('month'):
    base=g[g.model=='v7'].groupby('panel').iou.mean()
    new=pd.concat([p[p.model==f'ml_{choices[panel]}'] if panel in choices else p[p.model=='v7'] for panel,p in g.groupby('panel')]).groupby('panel').iou.mean()
    print(month,pd.DataFrame({'v7':base,'v8':new,'gain':new-base}).to_string())
    print('ongoing mean',base.mean(),new.mean())
for month,g in extended.groupby('month'):
    means=g.groupby('panel')[['v7','v8']].mean()
    means.loc[~means.index.isin(choices),'v8']=means.loc[~means.index.isin(choices),'v7']
    print('extended',month,means.mean().to_dict())
