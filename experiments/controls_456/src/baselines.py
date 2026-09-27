"""Same-feature comparison settings are selected on old ImageNet dev1 only, never on CUB/CIFAR."""
import json,sys,os
import numpy as np
import torch
from core import p_high,metrics
from data import ROOT,VENDOR,Bank,dump
sys.path.insert(0,str(VENDOR))
from vins.r5 import maha_pp
KS=[1,2,3,5,10,20,50];LAMS=[0.,.001,.01,.05,.1,.2,.5]

def knn(support,q,ks,device):
    S=torch.as_tensor(support.reshape(-1,support.shape[-1]),device=device,dtype=torch.float32)
    out={k:[] for k in ks}
    for lo in range(0,len(q),512):
        Q=torch.as_tensor(np.asarray(q[lo:lo+512]),device=device,dtype=torch.float32)
        vals=torch.topk(Q@S.T,max(ks),dim=1).values.cpu().numpy()
        for k in ks:out[k].append(1-vals[:,k-1].astype(np.float64))
    return {k:np.concatenate(v) for k,v in out.items()}

def model_scores(bank,name,device='cuda',ks=KS,lams=LAMS):
    F=bank.features[name];S=F[:bank.ns].reshape(bank.nclass,12,-1);c=slice(bank.ns,bank.ns+bank.nc)
    scores={}
    for k,d in knn(S,F,ks,device).items():scores[f'knn_{k}']=p_high(d[c],d)
    for lam,d in maha_pp(S,F,lams).items():scores[f'maha_{lam:g}']=p_high(d[c],d)
    return scores

def freeze_baselines(device='cuda'):
    out=ROOT/'baseline_selection.json'
    if out.exists():return
    b=Bank('dev').load(['B14','L14','CLIP']);f=b.frame
    I=np.flatnonzero(f.dataset=='id_dev');O=np.flatnonzero(f.dataset=='near_dev')
    allscores={n:model_scores(b,n,device) for n in ['B14','L14','CLIP']}
    chosen={};audit={}
    for names in [['B14'],['CLIP'],['B14','L14']]:
        tag='-'.join(names);chosen[tag]={};audit[tag]={}
        for family in ['knn','maha']:
            opts=[k for k in allscores[names[0]] if k.startswith(family+'_')]
            for k in opts:
                s=np.prod([allscores[n][k] for n in names],axis=0);audit[tag][k]=metrics(s[I],s[O])
            winner=min(opts,key=lambda k:(audit[tag][k]['FPR95'],-audit[tag][k]['AUROC'],opts.index(k)))
            chosen[tag][family]=winner
    dump(out,{'selection_data':'old ImageNet dev1 id_dev vs near_dev, static scores only','chosen':chosen,'metrics':audit,
              'new_holdout_data_used':False})

def selected_scores(bank,names,device):
    cfg=json.loads((ROOT/'baseline_selection.json').read_text())['chosen']['-'.join(names)]
    k=int(cfg['knn'].split('_')[1]);lam=float(cfg['maha'].split('_')[1]);allsc={}
    for n in names:allsc[n]=model_scores(bank,n,device,[k],[lam])
    return {'kNN':np.prod([allsc[n][cfg['knn']] for n in names],axis=0),
            'Mahalanobis++':np.prod([allsc[n][cfg['maha']] for n in names],axis=0)}
