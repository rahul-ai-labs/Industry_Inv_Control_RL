import os
import numpy as np
import torch
from torch import nn

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

def _enc(o):
    inv=np.asarray(o["inventory"],dtype=np.float32)
    pipe=np.asarray(o["arrival_pipeline"],dtype=np.float32)
    hist=np.asarray(o["demand_history"],dtype=np.float32)
    day=np.asarray(o["day"],dtype=np.float32).reshape(-1)
    util=np.asarray(o["capacity_utilisation"],dtype=np.float32).reshape(-1)
    dem=_demand(o); pos=inv+pipe.sum(axis=1); dos=pos/np.maximum(dem,1.)
    ld=dem*LEAD; gap=ld-pos; nxt=pipe[:,0]
    return np.concatenate([
        np.clip(inv/200.,0,3),np.clip(pipe.reshape(-1)/100.,0,5),
        np.clip(hist.reshape(-1)/80.,0,5),np.clip(day/50.,0,1),
        np.clip(util,0,1.5),np.clip(dem/60.,0,3),np.clip(pos/250.,0,4),
        np.clip(dos/5.,0,3),np.clip(ld/150.,0,3),np.clip(gap/150.,-3,3),
        np.clip(nxt/100.,0,5)]).astype(np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(56,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.actor=nn.Linear(256,33); self.critic=nn.Linear(256,1)
    def forward(self,x):
        h=self.body(x); return self.actor(h).view(-1,3,11),self.critic(h).squeeze(-1)

_net=Net()
_net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),"a2c.pt"),map_location="cpu",weights_only=True))
_net.eval()

def run_policy(observation):
    with torch.no_grad():
        logits,_=_net(torch.from_numpy(_enc(observation)).unsqueeze(0))
        action=logits.squeeze(0).argmax(dim=1).numpy()
    return (action*10).astype(int).tolist()
