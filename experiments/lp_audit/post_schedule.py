"""Run the remaining preregistered stages with a cumulative 8 GPU-hour hard stop."""
import os,json,subprocess,time,fcntl
from pathlib import Path
H=Path(__file__).resolve().parent;PY='/home/omote/granood_ke/.venv/bin/python'


def main():
    lock=(H/'reports/post_dispatch.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # The first dispatcher has a separate 4h limit. Do not race its GPUs.
    while True:
        p=H/'reports/primary_dispatch.json'
        if p.exists():
            d=json.loads(p.read_text())
            if 'active' not in d:break
        time.sleep(5)
    baseline=float(d['gpu_seconds_completed'])+200. # Conservative pilot/startup/TINS audit reservation.
    previous=H/'reports/post_dispatch.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('active'):raise RuntimeError('Existing active jobs must be accounted for before restarting the dispatcher')
        history=H/'reports/dispatch_history';history.mkdir(exist_ok=True)
        (history/f'post_{time.time_ns()}.json').write_text(json.dumps(old,indent=2))
        baseline=max(baseline,float(old['prior_gpu_seconds'])+float(old['gpu_seconds_completed']))
    reference_seconds=sum(x['seconds'] for x in d['completed'] if x.get('state')=='COMPLETE')/max(1,sum(x.get('state')=='COMPLETE' for x in d['completed']))
    tasks=[]
    for r in range(50):tasks.append(('real_calibration',f'rep{r}',[PY,'real_calibration.py','--rep',str(r)],H/f'results/real_calibration/rep{r}.json'))
    for kind in ['baselines','capacity','tuning']:
        for draw in range(5):
            for seed in [123,124,125]:
                for stream in ['near','far']:
                    name=f'draw{draw}_{stream}_seed{seed}'
                    tasks.append((kind,name,[PY,'alternatives.py','--kind',kind,'--draw',str(draw),'--stream',stream,'--seed',str(seed)],H/f'results/{kind}/{name}.json'))
    pending=[t for t in tasks if not t[3].exists()];active={};done=[];skipped=[];admitted=set();spent=0.;limit=8*3600
    ledger=H/'reports/post_dispatch.json'
    while pending or active:
        now=time.monotonic();inflight=sum(now-x['start'] for x in active.values())
        # Start stages only after the preceding stage finishes. Preserve complete
        # paired grids rather than spending the last budget on a partial grid.
        if pending and not active and pending[0][0] not in admitted:
            stage=pending[0][0];count=sum(t[0]==stage for t in pending)
            per_task=reference_seconds*{'real_calibration':.6,'baselines':.8,'capacity':1.,'tuning':2.2}[stage]
            estimate=count*per_task
            if baseline+spent+estimate>limit:
                skipped.extend(dict(stage=t[0],name=t[1],state='BUDGET_NOT_ADMITTED',
                                    stage_estimated_gpu_seconds=estimate,remaining_gpu_seconds=limit-baseline-spent)
                               for t in pending if t[0]==stage)
                pending=[t for t in pending if t[0]!=stage]
                print(json.dumps({'stage':stage,'state':'BUDGET_NOT_ADMITTED','estimated_seconds':estimate,'remaining_seconds':limit-baseline-spent}),flush=True)
                continue
            admitted.add(stage)
        if baseline+spent+inflight>=limit:
            for x in active.values():x['p'].terminate()
            for x in active.values():
                x['p'].wait();x['log'].close();done.append(dict(stage=x['task'][0],name=x['task'][1],state='BUDGET_STOP',seconds=time.monotonic()-x['start']))
            active={};break
        for gpu in range(4):
            if gpu in active or not pending:continue
            if pending[0][0] not in admitted:continue
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>1000:continue
            task=pending.pop(0);stage,name,cmd,target=task
            f=open(H/'logs'/f'{stage}_{name}.log','a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONWARNINGS':'ignore','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            p=subprocess.Popen(cmd,cwd=H,env=env,stdout=f,stderr=subprocess.STDOUT)
            active[gpu]={'task':task,'p':p,'start':time.monotonic(),'log':f}
        for gpu,x in list(active.items()):
            code=x['p'].poll()
            if code is None:continue
            dur=time.monotonic()-x['start'];spent+=dur;x['log'].close()
            done.append(dict(stage=x['task'][0],name=x['task'][1],state='COMPLETE' if code==0 else 'FAILED',exit_code=code,seconds=dur,gpu=gpu))
            print(json.dumps(done[-1]),flush=True);del active[gpu]
        ledger.write_text(json.dumps({'completed':done,'active':[{'gpu':g,'stage':x['task'][0],'name':x['task'][1],'elapsed':time.monotonic()-x['start']} for g,x in active.items()],
                                     'pending':[(x[0],x[1]) for x in pending],'skipped':skipped,'prior_gpu_seconds':baseline,'gpu_seconds_completed':spent,'hard_limit_gpu_seconds':limit},indent=2))
        time.sleep(3)
    ledger.write_text(json.dumps({'completed':done,'pending':[(x[0],x[1]) for x in pending],'skipped':skipped,
                                  'prior_gpu_seconds':baseline,'gpu_seconds_completed':sum(x['seconds'] for x in done),'hard_limit_gpu_seconds':limit},indent=2))


if __name__=='__main__':main()
