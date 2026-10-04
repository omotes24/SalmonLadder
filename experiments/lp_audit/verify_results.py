"""Audit completed files without refitting, selecting, or changing any score."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    checks=[];errors=[];manifest=[]
    for stage in ['primary','baselines','tuning','capacity']:
        for p in sorted((H/'results'/stage).glob('draw*.json')):
            info=json.loads(p.read_text());stem=p.stem;z=np.load(p.with_suffix('.npz'),allow_pickle=True)
            ids=z['sample_id'].astype(str);flag=z['is_ood'].astype(bool)
            if len(np.unique(ids))!=len(ids):errors.append(f'{stage}/{stem}: duplicate sample id')
            ref=H/'results/primary'/f'{stem}.npz'
            if stage!='primary' and ref.exists():
                a=np.load(ref,allow_pickle=True)
                for k in ['sample_id','is_ood','batch_index','S']:
                    if not np.array_equal(z[k],a[k]):errors.append(f'{stage}/{stem}: unmatched {k}')
            for m in info['metrics']:
                if not (0<=m['AUROC']<=100 and 0<=m['FPR95']<=100):errors.append(f'{stage}/{stem}: metric range')
                if m['actual_ID_acceptance']<95-1e-10:errors.append(f'{stage}/{stem}: ID acceptance below95')
            for k in z.files:
                if k in ['sample_id','is_ood','batch_index']:continue
                a=z[k]
                if np.issubdtype(a.dtype,np.number) and (np.isnan(a).any() or np.isposinf(a).any()):errors.append(f'{stage}/{stem}: invalid {k}')
            trace=H/'results'/stage/f'{stem}_{"solver" if stage=="primary" else "trace"}.csv'
            if trace.exists():
                t=pd.read_csv(trace)
                if 'solver' in t:
                    conv=t[t.solver.isin(['L2','class_L2'])]
                    if ((~conv.converged)|(conv.relative_residual>1e-8)).any():errors.append(f'{stage}/{stem}: convergence failure')
                if stage=='capacity':
                    if t.n_nodes.max()>14400+8192:errors.append(f'{stage}/{stem}: graph capacity violated')
                    if t.max_A1_A2_M.max()>8192:errors.append(f'{stage}/{stem}: memory capacity violated')
                if stage=='baselines':
                    bounded=t[t.capacity==8192]
                    if bounded.memory_size.max()>8192:errors.append(f'{stage}/{stem}: baseline capacity violated')
            checks.append(dict(stage=stage,run=stem,n_ID=int((~flag).sum()),n_OOD=int(flag.sum()),methods=len(info['metrics'])))
    # The metadata repair must preserve each registered calibration draw.
    from real_calibration import load_identities
    identity_path=H/'data/real_calibration/identities.npz'
    if identity_path.exists():
        identity=load_identities(identity_path)
        ready=json.loads((identity_path.parent/'ready.json').read_text())
        if sha(identity_path)!=ready['ids_sha256']:errors.append('real_calibration: preparation identity hash changed')
        fixed=set(identity['support_ids'])|set(identity['stream_ids'])
        for p in sorted((H/'results/real_calibration').glob('rep*.json')):
            info=json.loads(p.read_text());rep=int(info['rep'])
            rng=np.random.default_rng([972750,rep])
            pick=np.concatenate([np.flatnonzero(identity['pool_class']==c)[rng.choice(68,4,replace=False)] for c in range(900)])
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
                ids=z['calibration_ids']
                if not np.array_equal(ids,identity['pool_ids'][pick]):errors.append(f'real_calibration/rep{rep}: draw mismatch')
                if len(set(ids))!=3600 or set(ids)&fixed:errors.append(f'real_calibration/rep{rep}: overlap or duplicate')
                for k in z.files:
                    if k=='calibration_ids':continue
                    a=z[k];expected=4500 if '_ID_only_' in k else 6000
                    if len(a)!=expected or not np.isfinite(a).all() or (a<0).any() or (a>1).any():
                        errors.append(f'real_calibration/rep{rep}: invalid {k}')
            t=pd.read_csv(p.with_name(p.stem+'_solver.csv'))
            conv=t[t.solver=='L2']
            if ((~conv.converged)|(conv.relative_residual>1e-8)).any():errors.append(f'real_calibration/rep{rep}: convergence failure')
            checks.append(dict(stage='real_calibration',run=p.stem,n_calibration=3600,conditions=3,views=2))
    cfg=json.loads((H/'preregistration.json').read_text())
    for p in sorted((H/'results/synthetic').glob('rep*.json')):
        rep=int(p.stem[3:]);info=json.loads(p.read_text())
        if info['seed']!=cfg['simulation']['seed_base']+rep:errors.append(f'synthetic/rep{rep}: seed mismatch')
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
            for condition in cfg['simulation']['conditions']:
                flag=z[condition+'_is_ood']
                if len(flag)!=128 or int(flag.sum())!=(0 if condition=='iid_ID_only' else 32):
                    errors.append(f'synthetic/rep{rep}/{condition}: cohort mismatch')
            for k in z.files:
                if k.endswith(('_is_ood','_class')):continue
                a=z[k]
                if len(a)!=128 or not np.isfinite(a).all() or (a<0).any() or (a>1).any():errors.append(f'synthetic/rep{rep}: invalid {k}')
        checks.append(dict(stage='synthetic',run=p.stem,conditions=5))
    for p in sorted((H/'results').rglob('*')):
        if p.is_file():manifest.append(dict(path=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)))
    result={'checks':checks,'errors':errors,'result_files':len(manifest),
            'notes':'Checks apply only to completed receipts. Unexecuted/failed jobs remain in dispatch ledgers. Negative infinity is an exact log-zero score. Real calibration identities are replayed from the registered seeds and checked against the original preparation hash.'}
    (H/'reports/result_integrity.json').write_text(json.dumps(result,indent=2))
    (H/'reports/result_manifest.json').write_text(json.dumps(manifest,indent=2))
    cache_manifest=[dict(path=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p))
                    for p in sorted((H/'data').rglob('*')) if p.is_file()]
    (H/'reports/derived_cache_manifest.json').write_text(json.dumps(cache_manifest,indent=2))
    print(json.dumps({'checked_runs':len(checks),'result_files':len(manifest),'errors':errors}),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
