"""CPU-only final aggregation after the budget-controlled queue exits."""
import os,json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;PY='/home/omote/granood_ke/.venv/bin/python'
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
while True:
    try:
        ledger=json.loads((H/'reports/post_dispatch.json').read_text())
        if 'active' not in ledger:break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(10)
done=[]
for script in ['report.py','equivalence.py','analyze.py','write_reports.py','verify_results.py']:
    start=time.time()
    with (H/'logs/final_aggregation.log').open('a') as log:
        code=subprocess.run([PY,script],cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    done.append(dict(script=script,returncode=code,seconds=time.time()-start))
    (H/'reports/finalization_status.json').write_text(json.dumps({'steps':done,'complete':len(done)==5 and code==0},indent=2))
    if code:raise SystemExit(code)
