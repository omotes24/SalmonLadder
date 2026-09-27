"""Remove known train/test content overlap before first TINS inference; no metric tuning."""
from pathlib import Path
import json,hashlib,datetime,collections,io
from PIL import Image
R=Path(__file__).resolve().parent;M=R.parent
assert not (R/'launch.json').exists(),'Cannot revise inputs of an already launched run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
protocol=json.loads((R/'protocol.json').read_text())
rows=[json.loads(l) for l in (R/'prototype_manifest.jsonl').read_text().splitlines()]
evaluations=collections.defaultdict(list)
for line in (M/'manifest.jsonl').read_text().splitlines():
    row=json.loads(line);evaluations[row['sha256']].append(row['sample_id'])
initial=[row for row in rows if row['sha256'] in evaluations]
excluded=[{'path':r['path'],'sha256':r['sha256'],'evaluation_ids':evaluations[r['sha256']]} for r in initial]
before=sha(R/'prototype_manifest.jsonl');fixed=[];replacements=[]
for label in range(1000):
    original=[r for r in rows if r['label']==label];assert len(original)==16
    keep=[r for r in original if r['sha256'] not in evaluations]
    if len(keep)<16:
        folder=Path(original[0]['path']).parent;used={r['path'] for r in original}
        candidates=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
        for p in candidates:
            if str(p) in used:continue
            b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
            if digest in evaluations:
                excluded.append({'path':str(p),'sha256':digest,'evaluation_ids':evaluations[digest]});continue
            with Image.open(io.BytesIO(b)) as im:im.load()
            item={'path':str(p),'relative_path':str(p.relative_to(folder.parent.parent)),'label':label,'sha256':digest,'bytes':len(b)}
            keep.append(item);replacements.append(item)
            if len(keep)==16:break
    assert len(keep)==16
    fixed.extend(sorted(keep,key=lambda x:x['path']))
assert len(fixed)==16000 and all(r['sha256'] not in evaluations for r in fixed)
(R/'prototype_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fixed))
(R/'prototype_train16.txt').write_text(''.join(f'{r["relative_path"]} {r["label"]}\n' for r in fixed))
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_overlap_images':len(initial),
         'excluded':excluded,'replacements':replacements,'final_overlap':0,'per_class':16,
         'original_manifest_sha256':before,'final_manifest_sha256':sha(R/'prototype_manifest.jsonl'),
         'reason':'Known exact content duplicates of frozen test/calibration images were excluded from training prototypes before first TINS inference. No performance metrics used.',
         'deviation':'Official first-N fallback modified only to skip known content duplicates; not the unpublished author-specific 16-shot sample.'}
write(R/'prototype_overlap_resolution.json',receipt)
protocol['initial_prototype_content_overlap']=protocol['prototype_content_overlap']
protocol['prototype_content_overlap']=[]
protocol['prototype_selection']='First 16 lexicographically ordered training images per class excluding exact content duplicates of frozen test/calibration images. Eight original candidates replaced before TINS inference. Author-specific 16-shot list unavailable.'
protocol['prototype_list_sha256']=sha(R/'prototype_train16.txt')
protocol['runner_sha256']=sha(R/'run.py')
protocol['finalized_before_first_TINS_inference_utc']=receipt['utc']
protocol['input_selection_deviation_receipt_sha256']=sha(R/'prototype_overlap_resolution.json')
write(R/'protocol.json',protocol)
audit=json.loads((R/'data_audit.json').read_text());audit['initial_prototype_overlap']=audit['prototype_overlap'];audit['prototype_overlap']=[]
audit['selection_deviation']='8 duplicate training images replaced by next eligible images; 16 per class maintained.'
write(R/'data_audit.json',audit)
print(json.dumps({'ready':True,'initial_overlap_images':len(initial),'final_overlap':0,'prototype_images':len(fixed),'runner_hash_matches':sha(R/'run.py')==protocol['runner_sha256']}))
