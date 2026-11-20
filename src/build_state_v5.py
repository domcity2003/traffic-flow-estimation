"""Stream bounded v3/v4 state blend with exact natural-key checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from itertools import zip_longest
from src.state_v5 import bounded_blend

def build(output=Path('outputs/v5/state.csv'), blend=.5):
    root=Path('data/release/corridors')
    lanes={p.name:pd.read_csv(p/'network/fd_parameters.csv').set_index('link_id').lanes for p in root.iterdir() if p.is_dir()}
    output.parent.mkdir(parents=True,exist_ok=True)
    keys=['panel','timestamp','station_id','link_id','mask_regime']
    first=True;count=0
    for base,candidate in zip_longest(pd.read_csv('outputs/v1/state.csv',chunksize=200000,dtype={k:str for k in keys}),pd.read_csv('outputs/v4/state.csv',chunksize=200000,dtype={k:str for k in keys})):
        if base is None or candidate is None or not base[keys].equals(candidate[keys]):
            raise ValueError('v3/v4 state key mismatch')
        lane=np.array([lanes[p].loc[l] for p,l in zip(base.panel,base.link_id)])
        base['speed_kmh']=bounded_blend(base.speed_kmh,candidate.speed_kmh,5.,blend=blend)
        base['flow_vph']=bounded_blend(base.flow_vph,candidate.flow_vph,50.*lane,blend=blend)
        base.to_csv(output,mode='w' if first else 'a',header=first,index=False)
        first=False;count+=len(base)
    print('Wrote',count,'state rows',flush=True)

if __name__=='__main__':build()
