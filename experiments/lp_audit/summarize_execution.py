"""Describe completion, failures and conservative budget accounting without rerunning experiments."""
import csv,json
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'reports'


def main():
    primary=json.loads((O/'primary_dispatch.json').read_text())
    post=json.loads((O/'post_dispatch.json').read_text())
    if 'active' in primary or 'active' in post:raise RuntimeError('Experiment dispatch is not final')
    history=[json.loads(p.read_text()) for p in sorted((O/'dispatch_history').glob('post_*.json'))]
    failed=[]
    for source,ledger in [('primary',primary),('post',post)]+[(f'history{i}',d) for i,d in enumerate(history)]:
        for row in ledger['completed']:
            if row['state']!='COMPLETE':failed.append(dict(ledger=source,**row))
    skipped=post['skipped']+[dict(stage=x[0],name=x[1],state='PENDING_AT_BUDGET_STOP') for x in post.get('pending',[])]
    # Primary and current successful jobs have measured dispatch durations.
    # Previous failed attempts include documented conservative upper bounds.
    measured_success=sum(x['seconds'] for d in [primary,post] for x in d['completed'] if x['state']=='COMPLETE')
    charged=post['prior_gpu_seconds']+post['gpu_seconds_completed']
    receipts=json.loads((O/'runtime_receipts.json').read_text())
    primary_receipts=[r for r in receipts if r['stage']=='primary']
    summary={
        'completion':json.loads((O/'completion_status.json').read_text()),
        'final_dispatch_active':False,
        'hard_limit_gpu_hours':8,
        'charged_gpu_hours_conservative':charged/3600,
        'measured_successful_dispatch_gpu_hours':measured_success/3600,
        'pilot_and_startup_allowance_gpu_seconds':200,
        'failed_attempts':failed,
        'unexecuted_individual_tasks':skipped,
        'unexecuted_counts':{str(k):v for k,v in Counter((r['stage'],r['state']) for r in skipped).items()},
        'primary_peak_cpu_GiB':max(r['cpu_peak_bytes'] for r in primary_receipts)/2**30,
        'primary_peak_gpu_allocated_GiB':max(r['gpu_allocated_peak'] for r in primary_receipts)/2**30,
        'primary_peak_gpu_reserved_GiB':max(r['gpu_reserved_peak'] for r in primary_receipts)/2**30,
        'cost_scope':'Allocated process wall time summed over GPUs. A conservative bound, not device active-kernel time. Previous failure bounds include paused-supervisor time. Cached features are reused; no end-to-end feature-extraction benchmark or monetary electricity estimate.',
        'new_holdout_opened':False,
        'selection_data':'previously used dev1 only',
        'out_of_scope':['Phase3 controlled recurrence','Phase4 operational rates and end-to-end costs','Phase5 new holdout and confirmatory Holm tests'],
        'additional_limitations':['Two extra LP/REPRISE graph configurations require the complete tuning stage; any budget-skipped stage remains unexecuted.',
                                  'AdaNeg_type and OODD_type are explicitly simplified surrogates, not official method reproductions.',
                                  'Real-feature ID distribution shift was not run; synthetic shift was run.']
    }
    if charged>8*3600+12:raise RuntimeError('Conservative charged total exceeded budget tolerance')
    (O/'execution_summary.json').write_text(json.dumps(summary,indent=2))
    columns=['stage','name','state','stage_estimated_gpu_seconds','remaining_gpu_seconds']
    with (O/'unexecuted_runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(skipped)
    print(json.dumps({k:summary[k] for k in ['completion','charged_gpu_hours_conservative','measured_successful_dispatch_gpu_hours','unexecuted_counts']}),flush=True)


if __name__=='__main__':main()
