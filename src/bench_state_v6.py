"""Evaluate bounded v5 corrections against eligible train truth."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.bench_state import template_month
from src.state import interpolate_grid
from src.state_v4 import complete_grid


def run(month, output):
    root = Path('data/release')
    config = json.loads(Path('config/state_weights.json').read_text())
    rows = []
    for panel, weights in config.items():
        p = root/'corridors'/panel
        order = pd.read_csv(p/'network/lwr_mainline_topology.csv').sort_values('order_index').mainline_link_id.astype(str).tolist()
        lanes = pd.read_csv(p/'network/fd_parameters.csv').set_index('link_id').lanes
        targets = template_month(root/'task1'/panel/'train/sample_submission_state.csv', month)
        totals = {}
        for path in sorted((p/'train/mainline_states_masked').glob(f'**/*{month.replace("-", "_")}*.parquet')):
            date = path.stem.removeprefix('synthetic_mainline_').replace('_','-')
            t = targets[targets.timestamp.str.startswith(date)]
            if t.empty: continue
            masked = pd.read_parquet(path)
            truthpath = next((p/'train/mainline_states').glob('**/'+path.name))
            truth = pd.read_parquet(truthpath)
            truth = t.merge(truth, on=['timestamp','station_id','link_id'], how='left')
            eligible = truth.is_score_eligible.fillna(False).astype(bool).to_numpy()
            for field in weights:
                frame = masked.pivot(index='timestamp',columns='link_id',values=field).reindex(columns=order)
                grid = frame.to_numpy(float)
                ti = frame.index.get_indexer(t.timestamp)
                li = pd.Index(order).get_indexer(t.link_id)
                temp, spatial = interpolate_grid(grid,0), interpolate_grid(grid,1)
                base = np.where(np.isfinite(temp)&np.isfinite(spatial), weights[field]*temp+(1-weights[field])*spatial,np.where(np.isfinite(temp),temp,spatial))
                candidates = {'base':base}
                rule = json.loads(Path('config/state_v4.json').read_text())[panel][field]
                lowrank = complete_grid(grid,rule['rank'])
                candidate = (1-rule['blend'])*base+rule['blend']*lowrank
                cap = 5. if field == 'speed_kmh' else 50.*lanes.reindex(order).to_numpy()[None,:]
                candidates['v4'] = candidate
                candidates['v5'] = base + .5*np.clip(candidate-base,-cap,cap)
                candidates['v6'] = base + .75*np.clip(candidate-base,-cap,cap)
                y = truth[field].to_numpy(float)
                valid = eligible & np.isfinite(y)
                divisor = truth.link_id.map(lanes).to_numpy(float) if field=='flow_vph' else np.ones(len(t))
                for name, pred in candidates.items():
                    key=(t.mask_regime.iloc[0],field,name)
                    err=(pred[ti,li]-y)/divisor
                    s=totals.setdefault(key,[0.,0])
                    s[0]+=np.sum(np.square(np.nan_to_num(err[valid],nan=10000)))
                    s[1]+=valid.sum()
        for (regime,field,name),(ss,n) in totals.items():
            rows.append(dict(panel=panel,month=month,regime=regime,field=field,model=name,n=n,rmse=np.sqrt(ss/n)))
        pd.DataFrame(rows).to_csv(output,index=False)
        print(panel, 'done',flush=True)
    df=pd.DataFrame(rows)
    print(df.groupby(['field','model']).rmse.mean().to_string(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--month',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.month,a.output)
