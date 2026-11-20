"""Month-cross-fitted onset priors and final-step conditional classifiers.

Run lab before production. Only released history is used for test features.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from src.queue_gen_lab import PANELS, window_iou
from src.queue_eval import take
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import fit, predict, feature_mask
from src.queue_onset_prior import OnsetPrior
from src.build_queue_v9 import official_template

OUT = Path('outputs/lab/onset66')

def load(p, directory='qwin2'):
    z = dict(np.load(Path('outputs/lab') / directory / f'{p}.npz'))
    return take(z, z['cond'] == 'queue_onset')

def feats(tr, z, cross=False, rich=False):
    import src.queue_model as qm
    qm.RICH = bool(rich)
    if not cross:
        x,names=OnsetPrior(tr).features(z)
        return demand_features(x,names,z,rich=='competitive') if rich in ('demand','competitive') else (x,names)
    months = np.array([d[:7] for d in z['date']])
    train_months = np.array([d[:7] for d in tr['date']])
    out = None
    for month in np.unique(months):
        sel = months == month
        a, names = OnsetPrior(take(tr, train_months != month)).features(take(z, sel))
        if out is None:
            out = np.empty((len(months), *a.shape[1:]), np.float32)
        out[sel] = a
    return demand_features(out,names,z,rich=='competitive') if rich in ('demand','competitive') else (out,names)

def demand_features(x,names,z,competitive=False):
    from src.queue_model import ffill_steps, shift
    f,_=ffill_steps(z['flow']/z['cap'][None,None,:])
    o,_=ffill_steps(z['occ'])
    r,_=ffill_steps(z['speed']/z['cut'][None,None,:])
    extras={}
    for label,a in [('f',f),('o',o),('r',r)]:
        extras[label+'_mean']=np.nanmean(a,1)
        extras[label+'_slope']=np.nanmean(a[:,-3:],1)-np.nanmean(a[:,:3],1)
        extras[label+'_std']=np.nanstd(a,1)
        for lag in (2,3,4,8,12): extras[f'{label}_hist{lag}']=a[:,-lag]
        for k in (-4,-2,-1,1,2,4):
            extras[f'{label}_trend{k}']=shift(extras[label+'_slope'],k)
            extras[f'{label}_now{k}']=shift(a[:,-1],k)
    extras['cap']=np.broadcast_to(z['cap'][None,:]/5000,f.shape[::2])
    extras['cut']=np.broadcast_to(z['cut'][None,:]/100,f.shape[::2])
    if competitive:
        candidate=x[:,5,:,names.index('prior_all')]>.01
        for label,a,sign in [('speed',r[:,-1],-1),('demand',f[:,-1],1),
                           ('occupancy',o[:,-1],1),('demand_trend',extras['f_slope'],1),
                           ('speed_trend',extras['r_slope'],-1)]:
            b=np.where(candidate & np.isfinite(a),sign*a,-np.inf)
            order=np.argsort(-b,axis=1)
            rank=np.argsort(order,axis=1).astype(np.float32)
            best=np.max(b,axis=1,keepdims=True)
            best=np.where(np.isfinite(best),best,0)
            extras[label+'_candidate_rank']=np.where(candidate,rank,100)
            extras[label+'_candidate_margin']=sign*a-best
    a=np.stack(list(extras.values()),-1)
    a=np.broadcast_to(a[:,None],(*x.shape[:3],a.shape[-1]))
    return np.concatenate([x,a],-1),names+list(extras)

def model(kind, seed=0):
    if kind == 'hgb':
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=.06,
            max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=3, random_state=seed)
    return LGBMClassifier(n_estimators=450, learning_rate=.035, num_leaves=23,
        min_child_samples=60, reg_lambda=5, colsample_bytree=.85,
        n_jobs=4, verbosity=-1, random_state=seed)

def pooled_train(tr, cross, final, kind='hgb', rich=False, seed=0):
    xx, yy, ww = [], [], []
    for p in PANELS:
        x, names = feats(tr[p], tr[p], cross, rich)
        e, y = tr[p]['elig'], tr[p]['truth']
        if final:
            x, e, y = x[:, 5], e[:, 5], y[:, 5]
        xx.append(x[e]); yy.append(y[e])
        # Corridors receive equal total weight; each window equal weight.
        w = np.broadcast_to((1 / np.maximum(e.sum(tuple(range(1,e.ndim))),1)).reshape((-1,) + (1,)*(e.ndim-1)), e.shape)
        ww.append(w[e] / len(e))
    cols = feature_mask(names, ('link','step') if final else ('link',))
    x, y, w = np.vstack(xx)[:,cols], np.concatenate(yy), np.concatenate(ww)
    m = model(kind, seed)
    m.fit(x, y, sample_weight=w/w.mean() if final else None)
    return m, cols

def pooled_predict(m, cols, tr, z, final, rich=False):
    x, _ = feats(tr, z, False, rich)
    if final:
        x = x[:,5]
    shape = x.shape[:-1]
    pr = m.predict_proba(x.reshape(-1,x.shape[-1])[:,cols])[:,1].reshape(shape)
    if final:
        out = np.zeros((len(pr),6,pr.shape[1]),np.float32)
        out[:,5] = pr
        return out
    return pr

def lab():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = {p: take(load(p), (load(p)['date'] >= '2030-07-01') & (load(p)['date'] < '2031-01-01')) for p in PANELS}
    tests = {p: {'june': take(official_arrays(p,'train'), official_arrays(p,'train')['cond']=='queue_onset'),
                 'janfeb':load(p,'qoff360_janfeb')} for p in PANELS}
    per = {p:fit(tr[p])[0] for p in PANELS}
    pp = {(p,s):predict(per[p],z) for p in PANELS for s,z in tests[p].items()}
    rows=[]
    variants=[('baseline',False,False,'hgb',False),('crossfit',True,False,'hgb',False),
              ('final_hgb',True,True,'hgb',False),('final_lgb',True,True,'lgb',False),
              ('final_rich',True,True,'lgb',True),('final_demand',True,True,'lgb','demand')]
    for name,cross,final,kind,rich in variants:
        m,cols=pooled_train(tr,cross,final,kind,rich)
        joblib.dump((m,cols),OUT/f'{name}.joblib')
        for p in PANELS:
            for s,z in tests[p].items():
                prob=pooled_predict(m,cols,tr[p],z,final,rich)
                np.save(OUT/f'{name}_{p}_{s}.npy',prob)
                for perw in (0,.2):
                    pr=(1-perw)*prob+perw*pp[p,s]
                    pred=choose_set(pr,z['elig'],'eiou')
                    iou=window_iou(pred,z['truth'],z['elig'])
                    rows.append(dict(model=name,perw=perw,panel=p,set=s,n=len(iou),iou=iou.mean()))
        d=pd.DataFrame(rows)
        d.to_csv(OUT/'lab.csv',index=False)
        print(d.groupby(['model','perw','set']).iou.mean().round(5).to_string(),flush=True)

def production(name,basew=.5):
    cfg={'crossfit':(True,False,'hgb',False),'final_hgb':(True,True,'hgb',False),
         'final_lgb':(True,True,'lgb',False),'final_rich':(True,True,'lgb',True),
         'final_demand':(True,True,'lgb','demand'),
         'official_hgb':(True,False,'hgb',False),
         'official_demand':(True,True,'lgb','demand'),
         'competitive':(True,True,'lgb','competitive')}[name]
    cross,final,kind,rich=cfg
    tr={p:official_train(p) if name.startswith('official') else load(p) for p in PANELS}
    OUT.mkdir(parents=True,exist_ok=True)
    ms=[pooled_train(tr,cross,final,kind,rich,seed) for seed in range(3 if kind=='lgb' else 1)]
    joblib.dump(ms,OUT/f'production_{name}.joblib')
    for p in PANELS:
        for split in ('validation','private'):
            o=official_arrays(p,split)
            oo=take(o,o['cond']=='queue_onset')
            prob=np.mean([pooled_predict(m,cols,tr[p],oo,final,rich) for m,cols in ms],0)
            np.save(OUT/f'production_{name}_{p}_{split}.npy',prob)
    assemble_saved(name,basew)
    print('production complete',name,flush=True)

def assemble_saved(name,basew):
    frames=[]
    for p in PANELS:
        for split in ('validation','private'):
            o=official_arrays(p,split); oo=take(o,o['cond']=='queue_onset')
            prob=np.load(OUT/f'production_{name}_{p}_{split}.npy')
            z=dict(np.load(Path('outputs/lab/probs17')/f'queue_onset_{p}_{split}.npz'))
            old=.2*z['per_plain'].astype(np.float32)+.8*z['pool_plain_60'].astype(np.float32)
            prob=(1-basew)*prob+basew*old
            pred=choose_set(prob,oo['elig'],'eiou')
            for j,i in enumerate(np.flatnonzero(o['cond']=='queue_onset')):
                t=official_template(p,split,o['window_id'][i])
                t['queue_pred']=pred[j,o['cell_step'][i],o['cell_link'][i]].astype(int)
                frames.append(t)
    pd.concat(frames).to_csv(OUT/f'{name}_w{basew}_queue.csv',index=False)

def official_train(p):
    z=load(p,'qoff60'); extra=load(p,'qoff360_janfeb')
    dynamic=('speed','flow','occ','truth','elig','cond','minute','date','dow')
    z.update({k:np.concatenate([z[k],extra[k]]) for k in dynamic})
    return z

def followup_lab():
    rows=[]
    for name,directory,final,kind,rich in [('final_demand','qwin2',True,'lgb','demand'),
            ('official_hgb','qoff60',False,'hgb',False),
            ('official_demand','qoff60',True,'lgb','demand')]:
        tr={}
        for p in PANELS:
            z=load(p,directory)
            tr[p]=take(z,(z['date']>='2030-07-01') & (z['date']<'2031-01-01'))
        m,cols=pooled_train(tr,True,final,kind,rich)
        joblib.dump((m,cols),OUT/f'{name}.joblib')
        for p in PANELS:
            o=official_arrays(p,'train')
            tests={'june':take(o,o['cond']=='queue_onset'),'janfeb':load(p,'qoff360_janfeb')}
            for s,z in tests.items():
                prob=pooled_predict(m,cols,tr[p],z,final,rich)
                np.save(OUT/f'{name}_{p}_{s}.npy',prob)
                for basew in (0,.5):
                    base=np.load(OUT/f'baseline_{p}_{s}.npy')
                    pr=(1-basew)*prob+basew*base
                    iou=window_iou(choose_set(pr,z['elig'],'eiou'),z['truth'],z['elig'])
                    rows.append(dict(model=name,basew=basew,panel=p,set=s,n=len(iou),iou=iou.mean()))
        d=pd.DataFrame(rows);d.to_csv(OUT/'followup_lab.csv',index=False)
        print(d.groupby(['model','basew','set']).iou.mean().round(5).to_string(),flush=True)

def competitive_lab():
    tr={}
    for p in PANELS:
        z=load(p);tr[p]=take(z,(z['date']>='2030-07-01') & (z['date']<'2031-01-01'))
    m,cols=pooled_train(tr,True,True,'lgb','competitive')
    rows=[]
    for p in PANELS:
        o=official_arrays(p,'train')
        for s,z in {'june':take(o,o['cond']=='queue_onset'),'janfeb':load(p,'qoff360_janfeb')}.items():
            prob=pooled_predict(m,cols,tr[p],z,True,'competitive')
            np.save(OUT/f'competitive_{p}_{s}.npy',prob)
            for basew in (0,.5):
                pr=(1-basew)*prob+basew*np.load(OUT/f'baseline_{p}_{s}.npy')
                iou=window_iou(choose_set(pr,z['elig'],'eiou'),z['truth'],z['elig'])
                rows.append(dict(model='competitive',basew=basew,panel=p,set=s,n=len(iou),iou=iou.mean()))
    d=pd.DataFrame(rows);d.to_csv(OUT/'competitive_lab.csv',index=False)
    print(d.groupby(['basew','set']).iou.mean().round(5).to_string(),flush=True)

def ensemble(mode,basew=.5):
    names=['final_demand','competitive','official_demand']
    if mode=='production':
        for p in PANELS:
            for s in ('validation','private'):
                prob=np.mean([np.load(OUT/f'production_{n}_{p}_{s}.npy') for n in names],0)
                np.save(OUT/f'production_ensemble_{p}_{s}.npy',prob)
        assemble_saved('ensemble',basew)
        return
    rows=[]
    for p in PANELS:
        o=official_arrays(p,'train')
        for s,z in {'june':take(o,o['cond']=='queue_onset'),'janfeb':load(p,'qoff360_janfeb')}.items():
            prob=np.mean([np.load(OUT/f'{n}_{p}_{s}.npy') for n in names],0)
            for w in (0,.5):
                pr=(1-w)*prob+w*np.load(OUT/f'baseline_{p}_{s}.npy')
                iou=window_iou(choose_set(pr,z['elig'],'eiou'),z['truth'],z['elig'])
                rows.append(dict(model='ensemble',basew=w,panel=p,set=s,n=len(iou),iou=iou.mean()))
    d=pd.DataFrame(rows);d.to_csv(OUT/'ensemble_lab.csv',index=False)
    print(d.groupby(['basew','set']).iou.mean().round(5).to_string(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--basew',type=float,default=.5); ap.add_argument('--name'); args=ap.parse_args()
    if args.mode=='lab':lab()
    elif args.mode=='followup_lab':followup_lab()
    elif args.mode=='competitive_lab':competitive_lab()
    elif args.mode=='ensemble_lab':ensemble('lab')
    elif args.mode=='ensemble':ensemble('production',args.basew)
    elif args.mode=='assemble':assemble_saved(args.name,args.basew)
    else:production(args.mode,args.basew)
