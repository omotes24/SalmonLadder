"""One-time accounting repair after the controller was stopped for an ID-cache bug."""
import json,time,shutil,subprocess
from pathlib import Path
H=Path('/home/omote/reprise_lp_audit_20260927')
live=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        cmd=(proc/'cmdline').read_text().replace('\0',' ')
        if cmd.startswith('/home/omote/granood_ke/.venv/bin/python') and ('real_calibration.py' in cmd or 'post_schedule.py' in cmd):live.append((proc.name,cmd))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not live,live
p=H/'reports/post_dispatch.json';stamp=p.stat().st_mtime;d=json.loads(p.read_text());now=time.time()
assert len(d['active'])==4 and len(d['completed'])==4
(H/'failures').mkdir(exist_ok=True)
for a in d['active']:
    log=H/'logs'/f"{a['stage']}_{a['name']}.log"
    assert 'Object arrays cannot be loaded' in log.read_text()
    start=stamp-a['elapsed'];upper=now-start;lower=log.stat().st_mtime-start
    d['completed'].append(dict(stage=a['stage'],name=a['name'],state='FAILED',exit_code=1,seconds=upper,gpu=a['gpu'],
        seconds_lower_bound=lower,seconds_upper_bound=upper,duration_status='upper bound charged; dispatcher paused after job started'))
for a in d['completed']:
    shutil.copy2(H/'logs'/f"{a['stage']}_{a['name']}.log",H/'failures'/f"attempt1_{a['stage']}_{a['name']}.log")
d['active']=[];d['gpu_seconds_completed']=sum(a['seconds'] for a in d['completed'])
d['interruption']='Metadata-only object-ID serialization fix. All8 computed runs failed before saving. Logs preserved; no performance-dependent changes.'
p.write_text(json.dumps(d,indent=2))
print(json.dumps({'failed_attempts':len(d['completed']),'charged_upper_GPUh':d['gpu_seconds_completed']/3600,
                  'primary_plus_prior':d['prior_gpu_seconds']/3600}),flush=True)
with (H/'post_schedule.log').open('a') as f:
    child=subprocess.Popen(['/home/omote/granood_ke/.venv/bin/python','-u','post_schedule.py'],cwd=H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('restarted controller',child.pid,flush=True)
