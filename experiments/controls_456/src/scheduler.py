"""Persistent local process queue; never kills or modifies another experiment.
R5 reserves GPUs until its TINS GPU phase completes. Jobs use one idle GPU and two CPU threads.
"""
import os,sys,time,json,subprocess,argparse,traceback
from pathlib import Path
from data import ROOT,OLD,dump

def claim(key):
    p=ROOT/'claims'/(key+'.json');p.parent.mkdir(parents=True,exist_ok=True)
    try:
        fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(fd,json.dumps({'pid':os.getpid(),'time':time.time()}).encode());os.close(fd);return True
    except FileExistsError:
        try:
            prior=json.loads(p.read_text());os.kill(prior['pid'],0)
        except ProcessLookupError:
            try:p.unlink()
            except FileNotFoundError:pass
            return claim(key)
        return False

def done(key):return (ROOT/'queue_done'/f'{key}.json').exists() or (ROOT/'results'/f'{key}.json').exists()
def failed(key):return (ROOT/'failures'/f'{key}.json').exists()

def gpu_available(gpu):
    # The existing R5 launcher plans to use all four cards; do not race its setup phase.
    status=OLD/'r5/logs/pipeline.status'
    txt=status.read_text() if status.exists() else ''
    if 'start' in txt and 'tins runs done' not in txt and 'DONE' not in txt:
        # If R5 has explicitly failed and its jobs are gone, actual GPU occupancy is the arbiter.
        ps=subprocess.check_output(['ps','-u','omote','-o','args='],text=True)
        active=any(('r5_pipeline.sh' in l or 'scripts/r5_' in l) and 'ps -' not in l for l in ps.splitlines())
        if active:return False,'R5_reserved_GPU_phase'
    try:
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        mem,util=[int(x.strip()) for x in s.strip().split(',')]
        if mem>1024 or util>10:return False,'GPU_busy'
    except Exception as e:return False,repr(e)
    return True,'idle'

def initial_jobs():
    jobs=[]
    for bank in ['cifar','cub']:
        for model in ['CLIP','B14','L14']:
            jobs.append((f'encode_{bank}_{model}',[sys.executable,'-u','-c',f'from data import encode_new; encode_new({bank!r},{model!r})'],
                         (ROOT/f'data/{bank}/manifest_frozen.json').exists()))
    dev=(ROOT/'data/dev/ready.json').exists()
    jobs.append(('thresholds',[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--calibrate'],dev))
    jobs.append(('baselines',[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--baselines'],dev))
    return jobs

def task_ready(t):
    if not (ROOT/f'data/{t["bank"]}/ready.json').exists():return False
    if t['kind']=='holdout':return (ROOT/'baseline_selection.json').exists()
    if t['kind'] in ['stream','runtime','recurrence']:return (ROOT/'thresholds.json').exists()
    return True

def record_blocked_dependencies(tasks):
    def prep_failed(name):
        p=ROOT/'status'/f'prepare_{name}.json'
        return p.exists() and json.loads(p.read_text()).get('phase')=='failed'
    for key,cmd,ready in initial_jobs():
        dep=key.split('_')[1] if key.startswith('encode_') else 'dev'
        if prep_failed(dep) and not done(key) and not failed(key):
            dump(ROOT/'failures'/f'{key}.json',{'job':key,'returncode':None,'reason':'preparation_failed','dependency':dep,'log':str(ROOT/'logs/prepare.log')})
    for t in tasks:
        key=t['id'];deps=['thresholds'] if t['kind']!='holdout' else ['baselines']+[f'encode_{t["bank"]}_{m}' for m in ['CLIP','B14','L14']]
        broken=[d for d in deps if failed(d)]
        if prep_failed(t['bank']):broken.append('prepare_'+t['bank'])
        if broken and not done(key) and not failed(key):
            dump(ROOT/'failures'/f'{key}.json',{'job':key,'returncode':None,'reason':'blocked_dependency','dependencies':broken,'log':str(ROOT/'logs/prepare.log')})

def run_job(key,cmd,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=ROOT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    dump(ROOT/'status'/f'worker_{gpu if gpu is not None else "cpu"}.json',{'phase':'running','job':key,'pid':os.getpid(),'gpu':gpu})
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=Path(__file__).parent,env=env,stdout=f,stderr=subprocess.STDOUT)
        dump(ROOT/'status'/f'process_{key}.json',{'pid':p.pid,'job':key,'gpu':gpu})
        rc=p.wait()
    target='queue_done' if rc==0 else 'failures'
    dump(ROOT/target/f'{key}.json',{'returncode':rc,'job':key,'gpu':gpu,'log':str(log),'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    return rc

def worker(gpu):
    while True:
        p=ROOT/'plans/tasks.json'
        tasks=json.loads(p.read_text()) if p.exists() else []
        record_blocked_dependencies(tasks)
        if tasks and all(done(t['id']) or failed(t['id']) for t in tasks):
            dump(ROOT/'status'/f'worker_{gpu}.json',{'phase':'complete','gpu':gpu});return
        avail,reason=gpu_available(gpu)
        if not avail:
            dump(ROOT/'status'/f'worker_{gpu}.json',{'phase':'waiting','reason':reason,'gpu':gpu,'pid':os.getpid()});time.sleep(30);continue
        picked=False
        for key,cmd,ready in initial_jobs():
            if ready and not done(key) and not failed(key) and claim(key):run_job(key,cmd,gpu);picked=True;break
        if picked:continue
        for task in tasks:
            key=task['id']
            if task_ready(task) and not done(key) and not failed(key) and claim(key):
                run_job(key,[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--task',key],gpu);picked=True;break
        if picked:continue
        if tasks and all(done(t['id']) or failed(t['id']) for t in tasks):
            dump(ROOT/'status'/f'worker_{gpu}.json',{'phase':'complete','gpu':gpu});return
        dump(ROOT/'status'/f'worker_{gpu}.json',{'phase':'waiting','reason':'dependencies_or_other_workers','gpu':gpu,'pid':os.getpid()});time.sleep(30)

def cpu_probe():
    p=ROOT/'plans/tasks.json'
    while not p.exists():time.sleep(15)
    for t in json.loads(p.read_text()):
        if t['kind']=='recurrence' and t['batch']==256 and t['solver']=='warm15' and t['r']==0:
            if not done(t['id']) and claim(t['id']):
                run_job(t['id'],[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--task',t['id'],'--device','cpu']);return

def cpu_baselines():
    while not (ROOT/'data/dev/ready.json').exists():time.sleep(15)
    if not done('baselines') and not failed('baselines') and claim('baselines'):
        run_job('baselines',[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--baselines','--device','cpu'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--cpu-probe',action='store_true');p.add_argument('--cpu-baselines',action='store_true');a=p.parse_args()
    if a.cpu_probe:cpu_probe()
    elif a.cpu_baselines:cpu_baselines()
    else:worker(a.gpu)

if __name__=='__main__':main()
