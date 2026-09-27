import os,sys,json,hashlib,subprocess,random,datetime,collections,io
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent;P=R/'repo';M=R.parent
PIN='194759d716b27534bf2e8eeb0d71f5c4f0dabc40'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def link(target,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():assert dest.resolve()==target.resolve(),str(dest)
    else:dest.symlink_to(target,target_is_directory=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()==PIN
assert not subprocess.check_output(['git','diff','--name-only'],cwd=P,text=True).strip()
os.chdir(P);sys.path.insert(0,str(P));import eval_tins_w_init as t
from torchvision.datasets.folder import IMG_EXTENSIONS
audit=json.loads((M/'data_audit.json').read_text())
assert sha(M/'manifest.jsonl')==audit['manifest_sha256']
rows=[json.loads(l) for l in (M/'manifest.jsonl').read_text().splitlines()]
by_id={r['sample_id']:r for r in rows};by_hash=collections.defaultdict(list)
for r in rows:by_hash[r['sha256']].append(r['sample_id'])
source=Path('/home/omote/openood_simple_sota/data/benchmark_imglist/imagenet')
benchmark=R/'openood/data/benchmark_imglist/imagenet';benchmark.mkdir(parents=True,exist_ok=True)
datasets={'id':'imagenet','ssb_hard':'ssb_hard','ninco':'ninco','inaturalist':'inaturalist','textures':'textures','openimageo':'openimage_o'}
test_lists={};stream_ids={};count_total=0
for name,ds in datasets.items():
    filename=t.OPENOOD_IMAGENET1K_IMGLISTS[name];src=source/filename;lines=src.read_text().splitlines()
    (benchmark/filename).write_bytes(src.read_bytes());resolved=[]
    data_root=R/'openood/data'/('images_classic' if name=='textures' else 'images_largescale')
    for i,line in enumerate(lines):
        rel,label=line.split();r=by_id[f'{ds}_{i:06d}']
        assert r['relative_path']==rel and int(label)==r['label']
        components=Path(rel).parts;physical=Path(r['path'])
        for _ in components[1:]:physical=physical.parent
        link(physical,data_root/components[0])
        actual=data_root/rel
        assert actual.is_file() and actual.resolve()==Path(r['path']).resolve(),str(actual)
        resolved.append(r['sample_id'])
    assert len(resolved)==sum(r['dataset']==ds for r in rows)
    stream_ids[name]=resolved
    test_lists[name]={'count':len(resolved),'sha256':sha(src),'file':filename}
    count_total+=len(resolved)
assert count_total==130908
train=Path('/home/omote/openood_simple_sota/data/images_largescale/imagenet_1k/train')
classes=sorted(p for p in train.iterdir() if p.is_dir());assert len(classes)==1000
class_index=json.loads((P/'data/ImageNet/imagenet_class_index.json').read_text())
class_names=np.load(P/'data/ImageNet/imagenet_class_clean.npy',allow_pickle=False);assert len(class_names)==1000
link(train.parent,R/'datasets/ImageNet')
selected=[];overlap=[];trainlist=[];tasks=[]
for label,folder in enumerate(classes):
    assert class_index[str(label)][0]==folder.name
    entries=list(os.scandir(folder))
    assert not any(p.is_dir() for p in entries),'Unexpected nested train layout'
    files=sorted(Path(p.path) for p in entries if p.is_file() and Path(p.name).suffix.lower() in IMG_EXTENSIONS)
    assert len(files)>=16
    tasks.extend((label,p) for p in files[:16])
def inspect_train(task):
    label,p=task;b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
    with Image.open(io.BytesIO(b)) as im:im.load()
    return {'path':str(p),'relative_path':str(p.relative_to(train.parent)),'label':label,'sha256':digest,'bytes':len(b)}
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,item in enumerate(pool.map(inspect_train,tasks),1):
        selected.append(item);trainlist.append(f'{item["relative_path"]} {item["label"]}')
        if item['sha256'] in by_hash:overlap.append({'train':item['path'],'evaluation_or_calibration_ids':by_hash[item['sha256']]})
        if i%1600==0:print('prototype audit',i,flush=True)
(R/'prototype_train16.txt').write_text('\n'.join(trainlist)+'\n')
(R/'prototype_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
maps=R/'stream_manifests';maps.mkdir(exist_ok=True);streams={}
for group,names in t.OPENOOD_IMAGENET1K_GROUPS.items():
    for name in names:
        order=[(0,i) for i in range(len(stream_ids['id']))]+[(1,i) for i in range(len(stream_ids[name]))]
        random.Random(123).shuffle(order)
        ids=np.array([stream_ids[name if isood else 'id'][i] for isood,i in order])
        flags=np.array([a for a,b in order],dtype=np.int32)
        path=maps/f'openood_{group}_{name}.npz';np.savez_compressed(path,sample_id=ids,is_ood=flags)
        streams[path.stem]={'events':len(ids),'sha256':sha(path)}
weight=R/'weights/ViT-B-16.pt';assert sha(weight)=='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
args=['--eval-protocol','openood_imagenet1k','--openood-root',str(R/'openood'),
      '--root-dir',str(R/'datasets'),'--train-imglist',str(R/'prototype_train16.txt'),
      '--wordnet-dir',str(P/'txtfiles'),'--cache-dir',str(R/'cache'),'--name','tins_official_b16_seed0',
      '--gpu','0','--CLIP_ckpt','ViT-B/16','--batch-size','256','--prototype-batch-size','256',
      '--text-batch-size','1000','--seed','0','--stream-seed','123','--inversion-steps','30',
      '--inversion-reg-lambda','0.3','--ood-threshold','0.3','--group-num','5','--ood-number','2000',
      '--extra-text-length','2000','--bank-buffer-size','2000','--use-buffer','--save-stream-scores']
sys.argv=['eval_tins_w_init.py',*args];parsed=t.process_args()
write(R/'arguments.json',args)
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=P).decode().split('\0')
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'TINS','role':'additional comparator requested after original test results were disclosed; not a retroactive change to the six-comparator primary endpoint',
 'repo':'https://github.com/zxk1212/tins','commit':PIN,'gpu_physical':0,'parameters':vars(parsed),
 'source_sha256':{n:sha(P/n) for n in tracked if n},'runner_sha256':sha(R/'run.py'),
 'checkpoint_sha256':sha(weight),'prototype_selection':'Official code fallback: first 16 lexicographically ordered training images per ImageFolder class, now materialized as an explicit fixed list. Author 16-shot list is not included in repository.',
 'prototype_images':len(selected),'prototype_list_sha256':sha(R/'prototype_train16.txt'),
 'prototype_content_overlap':overlap,'eval_lists':test_lists,'test_unique_list_entries':count_total,
 'streams':streams,'stream_events':sum(v['events'] for v in streams.values()),
 'scoring':'Official per-dataset shuffled ID+OOD stream, dynamic bank reset per dataset, current batch rescored after its unsupervised adaptation; original batch size 256 retained.',
 'temperature':'Official code uses pretrained CLIP exp(logit_scale). The --tau argument is unused in its score implementation; retained at default.',
 'no_test_hyperparameter_tuning':True,'validation_not_used':True,
 'metric_reporting':'Save official metrics and separately compute original study FPR95 convention including threshold ties. Adaptive ID scores differ between OOD streams.',
 'uncertainty':'Image-wise bootstrap of cached adaptive scores would condition on the learned stream; do not treat it as a rerun of adaptation or add it to original frozen success criterion.'}
write(R/'protocol.json',protocol)
import torch,torchvision,pandas,scipy,sklearn,transformers
write(R/'environment.json',{m.__name__:m.__version__ for m in [torch,torchvision,np,pandas,scipy,sklearn,transformers]})
write(R/'data_audit.json',{'utc':protocol['utc'],'prototype_n':16000,'per_class':16,'prototype_overlap':overlap,'test_n':count_total,'eval_lists':test_lists,'all_paths_match_original_manifest':True})
print(json.dumps({'prepared':True,'prototype_images':len(selected),'prototype_overlap_groups':len(overlap),'test_images':count_total,'stream_events':protocol['stream_events']}))
