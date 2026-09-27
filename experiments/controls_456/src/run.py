import os,sys,json,time,resource,traceback,argparse
from pathlib import Path
import numpy as np
import torch
from data import ROOT,Bank,dump
from core import Detector,score_methods,metrics,operating,view_queries

def resources():
    return {'max_CPU_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'max_GPU_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
            'max_GPU_reserved_MiB':torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else 0}

def progress(task,phase,**kwargs):
    r={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'task':task['id'],'phase':phase,**kwargs}
    print(json.dumps(r),flush=True);dump(ROOT/'status'/f'{task["id"]}.json',r)

def detectors(bank,task,device):
    return {n:Detector(bank.views[n],device,task.get('cap'),task['solver'],bank.base/f'fixed_graph_{n}.npz') for n in task['models']}

def run_history(bank,task,indices,device):
    ds=detectors(bank,task,device);batch=task['batch'];times=[]
    for lo in range(0,len(indices),batch):
        _,tm=score_methods(ds,bank.features,bank.views,indices[lo:lo+batch]);times.append(tm)
        if lo%max(batch,256)==0:progress(task,'history',done=min(lo+batch,len(indices)),total=len(indices))
    return ds,times

def recurrence(task,device):
    b=Bank(task['bank']).load(task['models']);p=np.load(task['plan']);history=p['history'];query=np.r_[p['id_eval'],p['queries']]
    before=time.perf_counter();ds,timing=run_history(b,task,history,device)
    conditions={'singleton':np.zeros(0,np.int64)}
    if task['batch']>1:
        peers=p['peers'][:task['batch']-1].copy();conditions['peer_unrelated']=peers
        same=peers.copy();count=min(5,len(same));same[:count]=p['own_peers'][:count];conditions['peer_same_class']=same
    arrays={};results={};audit=[]
    snapshot={n:(d.seen,d.graph.u.copy(),[a.copy() for a in d.ages]) for n,d in ds.items()}
    for condition,peers in conditions.items():
        vals={k:[] for k in ['static','memory','lp','full']}
        for j,q in enumerate(query):
            # No evaluation image other than q enters this clone. Companions are unscored, disjoint reserve images.
            ix=np.r_[q,peers];s,_=score_methods(ds,b.features,b.views,ix,probe=True)
            for k in vals:vals[k].append(s[k][0])
            if j%32==0:progress(task,'independent_probes',condition=condition,done=j+1,total=len(query))
        for n,d in ds.items():
            assert d.seen==snapshot[n][0] and np.array_equal(d.graph.u,snapshot[n][1])
            assert all(np.array_equal(a,old) for a,old in zip(d.ages,snapshot[n][2]))
        vals={k:np.asarray(v) for k,v in vals.items()}
        ni=len(p['id_eval']);results[condition]={k:metrics(v[:ni],v[ni:]) for k,v in vals.items()}
        arrays.update({f'{condition}__{k}':v for k,v in vals.items()})
        # Fixed old-dev thresholds additionally reveal ID harm hidden by re-estimating FPR95 thresholds.
        th=ROOT/'thresholds.json'
        if th.exists():
            thresholds=json.loads(th.read_text())['visual']
            for k,v in vals.items():
                results[condition][k]['ID_false_alarms_per_1000_fixed_dev']=1000*float(np.mean(v[:ni]<thresholds[k]))
                results[condition][k]['OOD_detection_rate_fixed_dev']=float(np.mean(v[ni:]<thresholds[k]))
    return {'metrics':results,'seconds':time.perf_counter()-before,'resources':resources(),'n_ID_eval':len(p['id_eval']),
            'n_OOD_eval':len(p['queries']),'history_N':len(history),'history_OOD':int(b.frame.is_ood.iloc[history].sum()),
            'matching_mean_abs_log_static_p':float(p['match_abs_logp_difference'].mean()),'state_restoration_verified':True},dict(query_rows=query,history_rows=history,**arrays)

def stream_run(task,device,indices=None,threshold=False):
    b=Bank(task['bank']).load(task['models']);f=b.frame
    runtime=task['kind']=='runtime'
    if indices is None:
        if task['kind']=='holdout':
            ix=np.flatnonzero((f.role=='eval')&f.eligible);indices=np.random.default_rng(task['seed']).permutation(ix)
        else:indices=np.load(task['plan'])['stream']
    indices=np.asarray(indices);assert len(indices)==len(np.unique(indices))
    t0=time.perf_counter();encoder=tins=None
    # TINS comparison is evaluated on GPU only, never approximated by a static cached score.
    if device!='cpu':
        from bridge import Encoder,Tins
        encoder=Encoder(task['models']+['CLIP'] if runtime else ['CLIP'])
        tins=Tins(b,encoder)
    from baselines import selected_scores
    extra=selected_scores(b,task['models'],device) if task['kind']=='holdout' else {}
    ds=detectors(b,task,device);setup_s=time.perf_counter()-t0
    clip=np.load(b.base/'CLIP.npy',mmap_mode='r');batch=task['batch'];records=[];stored={};differences=[]
    for lo in range(0,len(indices),batch):
        ix=indices[lo:lo+batch];tick=time.perf_counter();ft={};modeldelta={}
        if runtime:
            feats,ft=encoder.encode(f.path.iloc[ix].tolist())
            modeldelta={n:float(np.max(np.abs(feats[n]-b.features[n][ix]))) for n in task['models']}
            # Use the newly encoded features for all detector stages, not just for a timing surrogate.
            pos=tins.z['positive_features'].detach().float().cpu().numpy()
            cand=np.argsort(-(feats['CLIP']@pos.T),axis=1)[:,:5]
            tstatic=time.perf_counter();sc={k:np.ones(len(ix)) for k in ['static','memory','lp','full']};di={}
            for n in task['models']:
                dd,pp,pa=view_queries(b.views[n],feats[n],cand)
                # Compute static work separately, then the original online update.
                ts=time.perf_counter();rr=ds[n].step(feats[n],dd,pa);di[n]=dict(ds[n].last)
                sc['static']*=pp
                for k in ['memory','lp','full']:sc[k]*=rr[k]
            static_and_visual=time.perf_counter()-tstatic
            cf=feats['CLIP']
        else:
            sc,di=score_methods(ds,b.features,b.views,ix);cf=np.asarray(clip[ix]);static_and_visual=None
        tins_s=0.
        if tins is not None:
            tickt=time.perf_counter();S=tins.step(cf);tins_s=time.perf_counter()-tickt
            both={'TINS':S,**{'TINS_x_'+k:S*v for k,v in sc.items()}}
            sc.update(both)
        for k,v in extra.items():
            sc[k]=v[ix]
            if tins is not None:sc['TINS_x_'+k]=S*v[ix]
        for k,v in sc.items():stored.setdefault(k,[]).append(np.asarray(v))
        if torch.cuda.is_available():torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        records.append({'start':lo,'n':len(ix),'total_s':elapsed,'features':ft,'TINS_s':tins_s,'visual':di,'encoded_feature_max_difference':modeldelta})
        if any(d.cap is not None and (any(len(a)>d.cap for a in d.ages) or len(d.graph.X)-d.graph.nfix>d.cap) for d in ds.values()):raise AssertionError('retention bound exceeded')
        if lo%max(batch,1024)==0:progress(task,'stream',done=min(lo+batch,len(indices)),total=len(indices),last_batch_s=elapsed)
    scores={k:np.concatenate(v) for k,v in stored.items()};o=f.is_ood.iloc[indices].to_numpy(bool)
    met={k:metrics(v[~o],v[o]) for k,v in scores.items()}
    times=np.array([r['total_s'] for r in records]);counts=np.array([r['n'] for r in records]);late=np.array([r['start']>=.75*len(indices) for r in records])
    perimage_latency=np.repeat(times,counts)
    perf={'scope':'end_to_end_raw_images' if runtime else 'cached_features_only_not_end_to_end',
          'startup_models_state_and_baselines_s':setup_s,'stream_total_s':float(times.sum()),
          'mean_processing_ms_per_image':1000*float(times.sum()/len(indices)),
          'last_quarter_processing_ms_per_image':1000*float(times[late].sum()/counts[late].sum()) if late.any() else None,
          'batch_latency_p95_ms':1000*float(np.quantile(times,.95)),
          'per_image_completion_latency_p95_ms_batch_ready':1000*float(np.quantile(perimage_latency,.95)),
          'arrival_queueing_included':False,'resources':resources(),'batch_timing':records}
    out={'metrics':met,'timing':perf,'n_ID':int((~o).sum()),'n_OOD':int(o.sum()),'all_scores_finite':all(np.isfinite(x).all() for x in scores.values())}
    if not out['all_scores_finite']:raise ValueError('nonfinite score')
    tp=ROOT/'thresholds.json'
    if tp.exists() and not threshold and task['kind'] in ['stream','runtime']:
        th=json.loads(tp.read_text());th={**th['visual'],**th.get('with_tins',{})}
        valid={k:v for k,v in scores.items() if k in th}
        out['operating_fixed_dev']=operating(valid,o,f['class'].iloc[indices].values,th)
    return out,dict(rows=indices,is_ood=o,**scores)

def calibrate(device):
    """One frozen threshold per method, pooled old-dev ID scores, targeting <=10 alarms/1,000 on that dev sample."""
    if (ROOT/'thresholds.json').exists():return
    from data import OLD
    import pandas as pd
    b=Bank('dev');index={s:i for i,s in enumerate(b.frame.sample_id)};parts=[];runs=[]
    if device!='cpu':
        from bridge import Tins
        from vins.tins_dev import run_stream
        tt=Tins(b);F=np.load(b.base/'CLIP.npy',mmap_mode='r')[b.ns+b.nc:b.ns+b.nc+512].copy()
        tt.t.setup_seed(0);tt.reset();ours=np.concatenate([tt.step(F[i:i+256]) for i in range(0,len(F),256)])
        tt.t.setup_seed(0)
        ref=run_stream(tt.t,tt.args,tt.model,tt.z,torch.from_numpy(F))
        same=bool(np.array_equal(ours,ref));delta=float(np.max(abs(ours-ref)))
        dump(ROOT/'tests/tins_equivalence.json',{'bitwise_equal':same,'max_abs_difference':delta,'n':len(F),'data':'old dev only'})
        if not same:raise AssertionError(f'TINS state adapter differs from upstream: {delta}')
        del tt
        torch.cuda.empty_cache()
    for seed in [123,124,125]:
        fr=pd.read_parquet(OLD/f'runs/main/stream_near_seed{seed}.parquet');ix=np.array([index[s] for s in fr.sample_id])
        task={'kind':'calibration','id':f'calibration_{seed}','bank':'dev','models':['B14','L14'],'batch':256,'solver':'warm15','cap':None}
        out,ar=stream_run(task,device,ix,threshold=True);parts.append({k:v[~ar['is_ood']] for k,v in ar.items() if k not in ['rows','is_ood']});runs.append(out)
        save(task,out,ar)
    allkeys=parts[0];thresholds={}
    for k in allkeys:
        v=np.sort(np.concatenate([x[k] for x in parts]));thresholds[k]=float(v[int(np.floor(.01*len(v)))])
    dump(ROOT/'thresholds.json',{'source':'old dev1 near streams, seeds123/124/125; ID scores only, OOD not used to optimize cutoff',
                                'target_ID_false_alarm_per_1000':10,'strict_alarm_rule':'score < threshold',
                                'visual':{k:v for k,v in thresholds.items() if k in ['static','memory','lp','full']},
                                'with_tins':{k:v for k,v in thresholds.items() if k not in ['static','memory','lp','full']},
                                'fixed_across_prevalence_recurrence_pattern_and_capacity':True})

def save(task,result,arrays):
    dest=ROOT/'results';dest.mkdir(parents=True,exist_ok=True)
    tmp=dest/(task['id']+f'.{os.getpid()}.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,dest/(task['id']+'.npz'))
    dump(dest/(task['id']+'.json'),{'task':task,**result});progress(task,'complete',result=str(dest/(task['id']+'.json')))

def main():
    a=argparse.ArgumentParser();a.add_argument('--task');a.add_argument('--device',default='cuda');a.add_argument('--calibrate',action='store_true');a.add_argument('--baselines',action='store_true');args=a.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')));torch.backends.cuda.matmul.allow_tf32=False
    if args.calibrate:calibrate(args.device);return
    if args.baselines:
        from baselines import freeze_baselines
        freeze_baselines(args.device);return
    tasks=json.loads((ROOT/'plans/tasks.json').read_text());task=next(t for t in tasks if t['id']==args.task)
    if (ROOT/'results'/f'{task["id"]}.json').exists():return
    progress(task,'starting',device=args.device)
    try:
        result,arrays=recurrence(task,args.device) if task['kind']=='recurrence' else stream_run(task,args.device)
        save(task,result,arrays)
    except Exception as e:
        progress(task,'failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
