"""Read-only audit of the frozen server inputs; writes only into this experiment."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get('REPRISE_DATA_ROOT', '/home/omote/vins_gonogo_20260925'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def main():
    frames = {dev: pd.read_parquet(BASE / ('dev2/' if dev == 'dev2' else '') / 'splits/samples.parquet')
              for dev in ['dev1', 'dev2']}
    shots = pd.read_parquet(BASE / 'r5/shots/draws.parquet')
    splits = {'history': {'dev1': 'USED development and selection', 'dev2': 'USED development confirmation; not holdout',
                         'OpenOOD': 'USED five development/evaluation cycles', 'Four-OOD': 'USED',
                         'CUB': 'USED', 'CIFAR100': 'USED', 'SSB-CUB': 'USED',
                         'new_holdout': 'NOT IDENTIFIED / NOT OPENED'}, 'draws': {}, 'overlap': {}}
    inputs = []
    for dev, d in frames.items():
        splits[dev] = d.groupby('split').size().to_dict()
        ev = d[d.split.isin(['id_dev', 'near_dev', 'far_dev'])]
        classes = json.loads((BASE / ('dev2/' if dev == 'dev2' else '') / 'splits/id_classes.json').read_text())
        dc = shots[shots.idx_1k.isin([c['idx_1k'] for c in classes])]
        for draw in range(5):
            x = dc[dc.draw == draw]
            sup, cal = x[x.role == 'support'], x[x.role == 'calib']
            splits['draws'][f'{dev}/draw{draw}'] = {
                'support': sup.sample_id.tolist(), 'calibration': cal.sample_id.tolist(),
                'support_cal_hash_overlap': len(set(sup.sha256) & set(cal.sha256)),
                'shots_eval_hash_overlap': len(set(x.sha256) & set(ev.sha256)),
                'shots_eval_id_overlap': len(set(x.sample_id) & set(ev.sample_id)),
                'unique_shot_hashes': int(x.sha256.nunique()), 'count': len(x)}
        inputs += [BASE / ('dev2/' if dev == 'dev2' else '') / 'splits' / n
                   for n in ['samples.parquet', 'id_classes.json', 'heldout.json', 'build_info.json']]
    for split in ['support', 'calib', 'id_dev', 'near_dev', 'far_dev']:
        splits['overlap'][f'dev1_dev2_{split}'] = len(set(frames['dev1'].query('split == @split').sha256)
                                                     & set(frames['dev2'].query('split == @split').sha256))
    splits['draws_distinct_by_hash'] = len({tuple(sorted(shots[shots.draw == d].sha256)) for d in range(5)})
    save('split_manifest.json', splits)
    records = pd.concat([frames['dev1'][['sample_id','path','sha256']],
                         frames['dev2'][['sample_id','path','sha256']], shots[['sample_id','path','sha256']]]).drop_duplicates('path')
    records.to_json(HERE / 'provenance/image_records.jsonl', orient='records', lines=True)
    def verify(row):
        try:
            actual = sha(row.path)
            return {'path': row.path, 'expected': row.sha256, 'actual': actual} if actual != row.sha256 else None
        except OSError as e:
            return {'path': row.path, 'error': str(e)}
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        failures = [x for x in pool.map(verify, records.itertuples()) if x]
    inputs += [BASE / 'r5/shots/draws.parquet', BASE / 'r5/phase3/prereg_phase3.json']
    inputs += list((BASE / 'features').glob('*.pt'))
    inputs += list((BASE / 'r5/features').glob('shots.*.pt'))
    inputs += list((BASE / 'r5/dev1/tins').glob('draw[0-4]/*.npz'))
    inputs += list((BASE / 'r5/dev1/tins_setup').glob('draw[0-4].pt'))
    entries = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in inputs if p.exists()]
    sources = [{'path': str(p.relative_to(HERE)), 'sha256': sha(p)}
               for p in sorted((HERE/'vendor').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
    save('data_manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
         'unique_image_files_verified': len(records), 'hash_failures': failures, 'inputs': entries,
         'source_files': sources, 'limits': ['Byte-identical duplicates audited; transformed copies, capture sessions and pretraining contamination not certified.',
         'New holdout not opened. Dev2/test used history retained. Feature extraction cache is not a future-neighbor graph.']})
    print(json.dumps({'images': len(records), 'failures': len(failures), 'inputs': len(entries)}), flush=True)
    # Re-analysis of issued predictions only, never selecting a new configuration on test.
    import sys
    sys.path.insert(0, str(BASE/'tins'))
    from utils.detection_util import get_measures
    rows = []
    for p in sorted((BASE/'r5/phase3/openood').glob('*.npz')):
        z = np.load(p, allow_pickle=True)
        if 'score_v5' not in z.files:
            continue
        flag = z['is_ood'].astype(bool)
        for method, s in [('tins',z['S']), ('v5',z['score_v5']),
                          ('v5_lp',z['S']*z['v5_plp_B14']*z['v5_plp_L14'])]:
            au, _, fpr = get_measures(s[~flag], s[flag])
            reference = json.loads(p.with_suffix('.json').read_text())['metrics'][method]
            rows.append({'file': p.name, 'method': method, 'AUROC': 100*au, 'FPR95': 100*fpr,
                         'delta_AUROC':100*au-reference['AUROC'], 'delta_FPR95':100*fpr-reference['FPR95']})
    pd.DataFrame(rows).to_csv(HERE/'reports/recomputed_paper_table.csv', index=False)
    save('provenance/evaluation_history.json', {'event': 'Reanalysis of previously issued OpenOOD predictions',
         'files': sorted(set(x['file'] for x in rows)), 'new_config_selection': False,
         'utc':datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
