"""Bounded local dispatcher; never stops jobs belonging to another experiment."""
import argparse,json,os,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent
PY='/home/omote/granood_ke/.venv/bin/python'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit-hours',type=float,default=4.0);ap.add_argument('--classwise',action='store_true');a=ap.parse_args()
    tasks=[(d,s,seed) for d in range(5) for seed in [123,124,125] for s in ['near','far']]
    (H/'logs').mkdir(exist_ok=True);pending=[]
    for d,s,seed in tasks:
        if not (H/f'results/primary/draw{d}_{s}_seed{seed}.json').exists():pending.append((d,s,seed))
    active={};done=[];spent=0.;limit=a.limit_hours*3600
    ledger=H/'reports/primary_dispatch.json'
    while pending or active:
        now=time.monotonic();inflight=sum(now-v['start'] for v in active.values())
        if spent+inflight>=limit:
            for x in active.values():x['p'].terminate()
            for x in active.values():x['p'].wait();x['log'].close();done.append({'task':x['task'],'state':'BUDGET_STOP','seconds':time.monotonic()-x['start']})
            active={};break
        for gpu in range(4):
            if gpu in active or not pending:continue
            # Recheck actual free device before every launch; do not contend with other jobs.
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>1000:continue
            d,s,seed=pending.pop(0);task=f'draw{d}_{s}_seed{seed}'
            log=open(H/'logs'/f'{task}.log','a')
            cmd=[PY,str(H/'run.py'),'--draw',str(d),'--stream',s,'--seed',str(seed)]
            if a.classwise:cmd+=['--classwise']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONWARNINGS':'ignore','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
            p=subprocess.Popen(cmd,cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]={'p':p,'start':time.monotonic(),'task':task,'log':log}
        for gpu,x in list(active.items()):
            code=x['p'].poll()
            if code is None:continue
            dur=time.monotonic()-x['start'];spent+=dur;x['log'].close()
            done.append({'task':x['task'],'exit_code':code,'seconds':dur,'gpu':gpu,'state':'COMPLETE' if code==0 else 'FAILED'})
            print(json.dumps(done[-1]),flush=True);del active[gpu]
        ledger.write_text(json.dumps({'completed':done,'active':[{'gpu':g,'task':v['task'],'elapsed':time.monotonic()-v['start']} for g,v in active.items()],
                                     'pending':pending,'gpu_seconds_completed':spent,'limit_seconds':limit},indent=2))
        time.sleep(3)
    ledger.write_text(json.dumps({'completed':done,'pending':pending,'gpu_seconds_completed':sum(x['seconds'] for x in done),'limit_seconds':limit},indent=2))


if __name__=='__main__':main()
