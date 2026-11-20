"""History-only features for learned ongoing queue evolution."""
import numpy as np
import pandas as pd
from src.queue import predict_window
from src.queue_v2 import predict_ongoing_propagation
from src.queue_v3 import predict_ongoing_lookback


def features(history, targets, origin, order, fd):
    times=pd.to_datetime(history.timestamp,utc=True)
    target_times=pd.to_datetime(targets.timestamp,utc=True)
    origin=pd.Timestamp(origin)
    if history.empty or times.max()>=origin or (target_times<=origin).any():
        raise ValueError('history must precede forecast origin')
    frame=history.copy()
    frame['link_id']=frame.link_id.astype(str)
    field=fd.set_index('link_id').reindex(order)
    def grid(name):
        values=pd.to_numeric(frame[name],errors='coerce').where(frame.is_score_eligible.astype(bool))
        return frame.assign(value=values).pivot(index='timestamp',columns='link_id',values='value').reindex(columns=order)
    speed=grid('speed_kmh')/field.free_speed_kmh.to_numpy()
    filled=speed.ffill()
    last=filled.iloc[-1].to_numpy()
    feat=[last,speed.iloc[-1].to_numpy(),filled.iloc[-3].to_numpy(),filled.iloc[-6].to_numpy(),
          speed.tail(3).mean().to_numpy(),speed.tail(3).min().to_numpy(),speed.mean().to_numpy(),
          last-filled.bfill().iloc[0].to_numpy(),speed.notna().mean().to_numpy()]
    for offset in [-2,-1,1,2]:
        neighbor=np.full(len(order),np.nan)
        if offset<0:neighbor[-offset:]=last[:offset]
        else:neighbor[:-offset]=last[offset:]
        feat.append(neighbor)
    flow=grid('flow_vph')/field.capacity_vph.to_numpy()
    occupancy=grid('occupancy')
    feat.extend([flow.ffill().iloc[-1].to_numpy(),flow.tail(3).mean().to_numpy(),
                 occupancy.ffill().iloc[-1].to_numpy(),occupancy.tail(3).mean().to_numpy(),
                 np.arange(len(order))/max(1,len(order)-1)])
    context=np.column_stack(feat)
    li=pd.Index(order).get_indexer(targets.link_id.astype(str))
    if (li<0).any():raise ValueError('unknown target link')
    minute=origin.hour*60+origin.minute
    horizon=(target_times-origin).dt.total_seconds().to_numpy()/60.
    x=np.column_stack([context[li],horizon,np.full(len(targets),np.sin(2*np.pi*minute/1440)),
                       np.full(len(targets),np.cos(2*np.pi*minute/1440))])
    return x.astype(np.float32)


def baseline(history,targets,origin,panel,cutoffs,order,settings,v2,v3):
    scaled={l:v*settings[panel]['ongoing_cutoff_factor'] for l,v in cutoffs.items()}
    change=v3.get(panel,{})
    prop=change.get('ongoing_propagation',v2.get(panel,{}).get('ongoing_propagation'))
    if change:
        return predict_ongoing_lookback(history,targets,origin,scaled,lookback_steps=change['ongoing_lookback'],link_order=order,**(prop or {}))
    if prop:return predict_ongoing_propagation(history,targets,origin,scaled,order,**prop)
    return predict_window(history,targets,origin,'queue_ongoing',scaled,[])
