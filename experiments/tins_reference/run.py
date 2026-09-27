"""Call unchanged official TINS; add durable progress, IDs, finite checks and metrics."""
import os,sys,json,time,traceback,hashlib,math,subprocess,fcntl
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R/'repo'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def atomic(p,obj):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(p)
lock=(R/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
current={'state':'starting','utc':utc(),'pid':os.getpid(),'physical_gpu':0,'completed_streams':[]}
def status(**kwargs):
    current.update(kwargs);current.update(utc=utc());atomic(R/'status.json',current)
status()
try:
    protocol=json.loads((R/'protocol.json').read_text());assert sha(__file__)==protocol['runner_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=P,text=True).strip()
    for name,digest in protocol['source_sha256'].items():assert sha(P/name)==digest,name
    assert sha(R/'prototype_train16.txt')==protocol['prototype_list_sha256']
    assert sha(R/'weights/ViT-B-16.pt')==protocol['checkpoint_sha256']
    for record in protocol['eval_lists'].values():
        assert sha(R/'openood/data/benchmark_imglist/imagenet'/record['file'])==record['sha256']
    for name,record in protocol['streams'].items():
        assert sha(R/'stream_manifests'/f'{name}.npz')==record['sha256']
    os.chdir(P);sys.path.insert(0,str(P));import eval_tins_w_init as t
    import torch
    torch.set_num_threads(4)
    original_download=t.official_clip.clip._download
    t.official_clip.clip._download=lambda url,root=None:original_download(url,str(R/'weights'))
    progress=t.tqdm
    def observed_progress(iterable,*args,**kwargs):
        desc=kwargs.get('desc','');total=kwargs.get('total',len(iterable) if hasattr(iterable,'__len__') else None)
        start=time.monotonic();status(phase=desc,phase_done=0,phase_total=total)
        for i,item in enumerate(progress(iterable,*args,**kwargs),1):
            yield item
            if i==1 or i%5==0 or i==total:
                status(phase=desc,phase_done=i,phase_total=total,phase_elapsed_seconds=time.monotonic()-start,
                       cuda_peak_MiB=torch.cuda.max_memory_allocated()/1024**2)
    t.tqdm=observed_progress
    original_load=t.load_official_clip
    def checked_load(args):
        status(state='loading_model');model,preprocess=original_load(args)
        atomic(R/'model_receipt.json',{'utc':utc(),'checkpoint_sha256':sha(R/'weights/ViT-B-16.pt'),
            'cuda_device':torch.cuda.get_device_name(),'logit_scale':float(model.logit_scale.exp().item()),
            'model_dtype':str(model.dtype),'preprocess':repr(preprocess),'official_source_unchanged':True})
        status(state='running');return model,preprocess
    t.load_official_clip=checked_load
    original_features=t.load_or_cache_stream_features_and_gt
    def checked_features(*args,**kwargs):
        features,labels=original_features(*args,**kwargs)
        assert torch.isfinite(features).all(),'Nonfinite image features'
        return features,labels
    t.load_or_cache_stream_features_and_gt=checked_features
    original_score=t.compute_grouped_positive_score
    def checked_score(*args,**kwargs):
        result=original_score(*args,**kwargs)
        assert torch.isfinite(result).all(),'Nonfinite grouped scores'
        return result
    t.compute_grouped_positive_score=checked_score
    original_expand=t.maybe_expand_dynamic_bank
    def checked_expand(*args,**kwargs):
        result=original_expand(*args,**kwargs)
        for item in result:
            if isinstance(item,torch.Tensor):assert torch.isfinite(item).all(),'Nonfinite adaptive bank'
        return result
    t.maybe_expand_dynamic_bank=checked_expand
    metrics={};original_eval=t.eval_mixed_stream_one_ood
    def observed_eval(*args,**kwargs):
        name=kwargs['stream_name'];status(stream=name,phase='begin_stream')
        inside,outside=original_eval(*args,**kwargs)
        assert np.isfinite(inside).all() and np.isfinite(outside).all()
        a=kwargs['args'];raw=Path(a.log_directory)/f'stream_scores_{name}.npz'
        with np.load(raw) as z,np.load(R/'stream_manifests'/f'{name}.npz') as mapping:
            assert np.array_equal(z['is_ood'],mapping['is_ood'])
            assert np.array_equal(z['scores'][z['is_ood']==0],inside)
            assert np.array_equal(z['scores'][z['is_ood']==1],outside)
            np.savez_compressed(R/f'scores_{name}.npz',sample_id=mapping['sample_id'],is_ood=mapping['is_ood'],
                                ID_score=z['scores'],OOD_score=-z['scores'])
        from sklearn.metrics import roc_auc_score
        iid=-inside.astype(float);ood=-outside.astype(float);threshold=np.sort(iid)[math.ceil(.95*len(iid))-1]
        metrics[name]={'AUROC':float(roc_auc_score(np.r_[np.zeros(len(iid)),np.ones(len(ood))],np.r_[iid,ood])),
            'FPR95':float(np.mean(ood<=threshold)),'ID_threshold':float(threshold),'actual_ID_TPR':float(np.mean(iid<=threshold)),
            'n_ID':len(iid),'n_OOD':len(ood),'NaN':0,'Inf':0,
            'ID_threshold_ties':int(np.sum(iid==threshold)),'OOD_threshold_ties':int(np.sum(ood==threshold))}
        atomic(R/'point_results.json',{'utc':utc(),'status':'partial','results':metrics,'comparison_type':'additional adaptive baseline; not original primary endpoint'})
        status(completed_streams=list(metrics));return inside,outside
    t.eval_mixed_stream_one_ood=observed_eval
    sys.argv=[str(P/'eval_tins_w_init.py'),*json.loads((R/'arguments.json').read_text())]
    t.main()
    assert len(metrics)==5
    summary={}
    for group in ['nearood','farood']:
        names=[k for k in metrics if k.startswith('openood_'+group+'_')]
        summary[group]={metric:float(np.mean([metrics[k][metric] for k in names])) for metric in ['AUROC','FPR95']}
    atomic(R/'point_results.json',{'utc':utc(),'status':'complete','results':metrics,'means':summary,
        'comparison_type':'additional adaptive baseline; not original primary endpoint','official_report_directory':str(P/protocol['parameters']['log_directory'])})
    status(state='complete',phase='complete')
except BaseException:
    status(state='failed',error=traceback.format_exc());raise
