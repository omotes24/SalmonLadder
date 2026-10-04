"""Finish the written report after the running experiment/verification pipeline exits."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'reports'


def main(upstream_pid):
    lock=(O/'finish_reports.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=O/'report_ready.json';steps=[]
    def save(state,**kwargs):
        status.write_text(json.dumps(dict(state=state,pid=os.getpid(),upstream_pid=upstream_pid,steps=steps,**kwargs),indent=2))
    save('WAITING_FOR_VERIFIED_RESULTS')
    try:
        while True:
            p=O/'finalization_status.json'
            try:d=json.loads(p.read_text()) if p.exists() else {}
            except json.JSONDecodeError:d={}
            if d.get('complete'):break
            if any(x.get('returncode') for x in d.get('steps',[])):raise RuntimeError('Upstream aggregation failed; see finalization_status.json and logs/final_aggregation.log')
            try:os.kill(upstream_pid,0)
            except ProcessLookupError:raise RuntimeError('Upstream aggregation process exited before completion')
            time.sleep(10)
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        for script in ['summarize_execution.py','make_summary.py']:
            start=time.time();save('BUILDING_REPORT',script=script)
            with (H/'logs/finish_reports.log').open('a') as out:
                code=subprocess.run([sys.executable,script],cwd=H,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
            steps.append(dict(script=script,returncode=code,seconds=time.time()-start))
            if code:raise RuntimeError(f'{script} failed; see logs/finish_reports.log')
        save('COMPLETE',summary=str(H/'summary.md'))
    except BaseException as error:
        save('FAILED',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream-pid',type=int,required=True)
    main(p.parse_args().upstream_pid)
