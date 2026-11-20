"""Time-conditioned onset sets learned solely from released training events."""
import numpy as np
import pandas as pd


def choose_links(events, origin, bandwidth_minutes=120):
    if not events or bandwidth_minutes <= 0:
        raise ValueError('events and positive bandwidth required')
    ts=pd.Timestamp(origin)
    if any(pd.Timestamp(e['origin']) >= ts for e in events):
        raise ValueError('training events must precede prediction origin')
    minute=ts.hour*60+ts.minute
    train_minutes=np.array([pd.Timestamp(e['origin']).hour*60+pd.Timestamp(e['origin']).minute for e in events])
    distance=abs(train_minutes-minute);distance=np.minimum(distance,1440-distance)
    weights=np.exp(-.5*(distance/bandwidth_minutes)**2)+.02
    links=sorted({link for e in events for link in e['queued']})
    positive=np.array([[link in e['queued'] for link in links] for e in events],dtype=bool)
    eligible=np.array([[link in e['eligible'] for link in links] for e in events],dtype=bool)
    probability=(weights@positive)/np.maximum(weights@eligible,1e-9)
    order=np.argsort(-probability,kind='stable')
    best_score=-1.;best=[]
    for k in range(1,min(20,len(links))+1):
        chosen=order[:k]
        intersection=positive[:,chosen].sum(axis=1)
        union=positive.sum(axis=1)+eligible[:,chosen].sum(axis=1)-intersection
        score=np.average(intersection/np.maximum(union,1),weights=weights)
        if score>best_score:
            best_score=score;best=[links[i] for i in chosen]
    return best


def predict_onset(targets, events, origin, bandwidth_minutes=120):
    times=pd.to_datetime(targets.timestamp,utc=True)
    if len(times)==0 or times.min()<=pd.Timestamp(origin):
        raise ValueError('forecast targets must follow origin')
    links=choose_links(events,origin,bandwidth_minutes)
    out=targets.copy()
    out['queue_pred']=(out.link_id.astype(str).isin(links)&(times==times.max())).astype(int)
    return out
