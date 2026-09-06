import os
import numpy as np

REF=np.asarray([30.,25.,35.],dtype=np.float32)
LEAD=np.asarray([3.,2.,1.],dtype=np.float32)

def _demand(o):
    h=np.asarray(o["demand_history"],dtype=np.float32)
    w0=np.asarray([.05,.07,.10,.13,.17,.21,.27],dtype=np.float32)
    out=np.empty(3,dtype=np.float32)
    for p in range(3):
        v=h[:,p]; m=v>0
        if not np.any(m): out[p]=REF[p]; continue
        w=w0[m]; w=w/w.sum(); r=float(np.dot(v[m],w)); c=min(int(m.sum()),7)/7.
        out[p]=c*r+(1.-c)*REF[p]
    return np.maximum(out,1.)

def _state(o):
    inv=np.asarray(o["inventory"],dtype=np.float32)
    pipe=np.asarray(o["arrival_pipeline"],dtype=np.float32)
    dem=_demand(o); pos=inv+pipe.sum(axis=1)
    cover=pos/np.maximum(dem*LEAD,1.)
    cb=np.digitize(cover,[.5,.8,1.,1.25,1.6]).astype(int)
    db=np.digitize(dem,[22.5,30.,37.5,45.]).astype(int)
    day=int(np.asarray(o["day"]).reshape(-1)[0]); phase=min(max((day-1)//10,0),4)
    util=float(np.asarray(o["capacity_utilisation"]).reshape(-1)[0])
    ub=int(np.digitize([util],[.35,.55,.75,.9])[0])
    return tuple(cb.tolist()+db.tolist()+[phase,ub])

def _fallback(o):
    inv=np.asarray(o["inventory"],dtype=np.float32)
    pipe=np.asarray(o["arrival_pipeline"],dtype=np.float32)
    dem=_demand(o); pos=inv+pipe.sum(axis=1)
    target=dem*(LEAD+.75)
    needed=np.maximum(target-pos,0.)
    q=np.clip(np.ceil(needed/10.)*10.,0,100)
    return (q/10.).astype(np.int64)

_d=np.load(os.path.join(os.path.dirname(__file__),"tabular_q.npz"))
_table={tuple(map(int,s)):v for s,v in zip(_d["states"],_d["q_values"])}

def run_policy(observation):
    v=_table.get(_state(observation))
    action=_fallback(observation) if v is None else np.argmax(v,axis=1).astype(np.int64)
    return (action*10).astype(int).tolist()
