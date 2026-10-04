"""Re-execute cached-feature TINS on complete-batch prefixes; no ground truth passed."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'vendor'));BASE=Path('/home/omote/vins_gonogo_20260925')
from vins.tins_dev import import_tins,make_args,load_clip,setup_to_device,run_stream


def main():
    start=time.perf_counter();torch.set_num_threads(1);t=import_tins()
    args=make_args(t,H/'data/tins_prefix_cache','lp_audit_prefix',123)
    net,_=load_clip(t,args)
    setup=torch.load(BASE/'r5/dev1/tins_setup/draw0.pt',map_location='cpu')
    state=setup_to_device(setup)
    b=torch.load(BASE/'runs/main/stream_feats_near_seed123.pt',map_location='cpu')
    x=b['features'][:768]
    ys=[]
    for tag,features in [('prefix',x[:512]),('extended',x),('changed_future',torch.cat([x[:512],torch.roll(x[512:],7,dims=1)]))]:
        t.setup_seed(args.seed)
        rec=run_stream(t,args,net,state,features)
        ys.append(np.asarray(rec))
    existing=np.load(BASE/'r5/dev1/tins/draw0/near_seed123.npz',allow_pickle=True)['S_final'][:512]
    out={'prefix_extended_max_error':float(np.max(np.abs(ys[0]-ys[1][:512]))),
         'prefix_changed_future_max_error':float(np.max(np.abs(ys[0]-ys[2][:512]))),
         'cached_full_stream_max_error':float(np.max(np.abs(ys[0]-existing))),
         'n_prefix':512,'extended_n':768,'seconds':time.perf_counter()-start,
         'gpu_seconds':time.perf_counter()-start,'condition':'dev1 draw0 near seed123, complete batches; reset RNG each run; fresh internal bank each call'}
    np.savez_compressed(H/'reports/tins_prefix_scores.npz',prefix=ys[0],extended=ys[1],changed_future=ys[2],cached=existing)
    (H/'reports/tins_prefix.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    assert out['prefix_extended_max_error']==0 and out['prefix_changed_future_max_error']==0


if __name__=='__main__':main()
