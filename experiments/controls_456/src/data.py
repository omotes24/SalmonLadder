import os,sys,json,hashlib,time,tarfile,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from core import norm,static_view

ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/home/omote/vins_gonogo_20260925')
LARGE=Path('/home/omote/ood_large_best_20260923')
VENDOR=Path(__file__).resolve().parent/'vendor'
sys.path.insert(0,str(VENDOR))

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def loadpt(p):return torch.load(p,map_location='cpu',weights_only=False)

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def pixel_hash(p):
    with Image.open(p) as im:
        im=im.convert('RGB');h=hashlib.sha256(str(im.size).encode());h.update(im.tobytes());return h.hexdigest()

def audit_images(frame,out):
    """Deduplicate by decoded RGB pixels; retain all fixed shots but disallow matching evaluation rows."""
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    cache={}
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line);cache[r['path']]=r['pixel_sha256']
    from concurrent.futures import ThreadPoolExecutor
    missing=list(dict.fromkeys(p for p in frame.path if p not in cache))
    with out.open('a') as f,ThreadPoolExecutor(max_workers=4) as pool:
        for lo in range(0,len(missing),256):
            chunk=missing[lo:lo+256]
            for p,h in zip(chunk,pool.map(pixel_hash,chunk)):
                cache[p]=h;f.write(json.dumps({'path':p,'pixel_sha256':h})+'\n')
            f.flush()
            if lo%4096==0:print(json.dumps({'pixel_audit':str(out),'hashed':len(cache),'total':len(frame)}),flush=True)
    frame=frame.copy();frame['pixel_sha256']=[cache[p] for p in frame.path]
    fixed=set(frame.loc[frame.role.isin(['support','calib']),'pixel_sha256'])
    seen=set(fixed);valid=[]
    for r in frame.itertuples():
        if r.role in ['support','calib']:valid.append(True)
        else:
            good=r.pixel_sha256 not in seen;valid.append(good)
            if good:seen.add(r.pixel_sha256)
    frame['eligible']=valid
    return frame

def prepare_openood():
    out=ROOT/'data/openood';out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return
    print('prepare OpenOOD feature tables',flush=True)
    b=loadpt(OLD/'test_eval/results/features.pt');l=loadpt(OLD/'test_eval/results/features_vitl14.pt')
    paths=b['paths'];assert paths==l['paths']
    ids=sorted({s for ds in ['ssb_hard','ninco','inaturalist','textures','openimageo'] for s in np.load(OLD/f'test_runs/default/{ds}_seed123.npz')['sample_id'].tolist()})
    assert len(paths)==16000+len(ids)
    manifest={r['sample_id']:r for r in map(json.loads,(LARGE/'manifest.jsonl').read_text().splitlines())}
    rows=[]
    for i,p in enumerate(paths[:16000]):
        role='support' if i<12000 else 'calib';c=i//12 if i<12000 else (i-12000)//4
        rows.append({'sample_id':f'shot:{i}','path':p,'role':role,'dataset':'imagenet','class':str(c),'id_class':c,'is_ood':False})
    for sid,p in zip(ids,paths[16000:]):
        r=manifest[sid];ds=r['dataset'];parts=Path(r['relative_path']).parts
        assert os.path.realpath(p)==os.path.realpath(r['path']),f'feature/sample alignment mismatch: {sid}'
        if sid.startswith('imagenet'):c=str(r['label']);ds='imagenet'
        else:c=parts[1] if len(parts)>2 and parts[1]!='images' else sid
        rows.append({'sample_id':sid,'path':p,'role':'eval','dataset':ds,'class':c,'id_class':int(r['label']) if sid.startswith('imagenet') else -1,'is_ood':not sid.startswith('imagenet')})
    frame=pd.DataFrame(rows)
    # The controlled studies need near-OOD + ID, not all far datasets. Keep original feature indices here.
    frame['original_row']=np.arange(len(frame))
    keep=(frame.role!='eval')|frame.dataset.isin(['imagenet','ninco','ssb_hard'])
    frame=frame[keep].reset_index(drop=True)
    frame=audit_images(frame,out/'pixel_hashes.jsonl')
    frame.to_parquet(out/'images.parquet',index=False)
    select=frame.original_row.values
    np.save(out/'B14.npy',b['dino'].numpy()[select].astype(np.float32));np.save(out/'L14.npy',l['dino'].numpy()[select].astype(np.float32))
    # Candidate rows in the old file begin with calibration; support candidates are not queried.
    cand=np.zeros((len(frame),5),dtype=np.int64)
    cand[12000:]=np.asarray(b['cand'])[select[12000:]-12000]
    np.save(out/'cand.npy',cand)
    clip=np.zeros((len(frame),512),np.float32)
    clip[12000:]=b['clip_q'].numpy()[select[12000:]-12000]
    # Original 16-shot sets are identical in dev1/dev2, whose ID class union covers ImageNet-1K.
    path_f={}
    for dev in [OLD,OLD/'dev2']:
        sm=pd.read_parquet(dev/'splits/samples.parquet');fb=loadpt(dev/'features/clip.pt')
        mp={s:i for i,s in enumerate(fb['sample_id'])}
        for rr in sm[sm.split.isin(['support','calib'])].itertuples():
            path_f[os.path.realpath(rr.path)]=fb['features'][mp[rr.sample_id]].numpy()
    for i,p in enumerate(frame.path[:12000]):clip[i]=path_f[os.path.realpath(p)]
    np.save(out/'CLIP.npy',clip)
    dump(out/'ready.json',{'n':len(frame),'excluded_eval_pixel_duplicates':int((~frame.eligible).sum()),'ns':12000,'nc':4000,'nclass':1000,'source':'frozen REPRISE caches','pretraining_unseen_claim':False})

def prepare_dev():
    out=ROOT/'data/dev';out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():return
    sm=pd.read_parquet(OLD/'splits/samples.parquet')
    sup=sm[sm.split=='support'].sort_values('class_idx_id',kind='stable')
    cal=sm[sm.split=='calib'].sort_values('class_idx_id',kind='stable')
    tail=sm[sm.split.isin(['id_dev','near_dev','far_dev'])]
    f=pd.concat([sup,cal,tail]).reset_index(drop=True)
    f['role']=['support']*len(sup)+['calib']*len(cal)+['eval']*len(tail)
    f['class']=f.wnid.fillna(f.sample_id).astype(str);f['is_ood']=f.split.isin(['near_dev','far_dev']);f['eligible']=True
    f['dataset']=f.split;f['id_class']=f.class_idx_id.astype(int)
    f.to_parquet(out/'images.parquet',index=False)
    for name,fn in [('B14','dino.pt'),('L14','dino_vitl14.pt'),('CLIP','clip.pt')]:
        b=loadpt(OLD/'features'/fn);mp={s:i for i,s in enumerate(b['sample_id'])}
        np.save(out/f'{name}.npy',b['features'][[mp[s] for s in f.sample_id]].numpy().astype(np.float32))
    dv=pd.read_parquet(OLD/'runs/dview/dview.parquet').set_index('sample_id')
    cand=np.zeros((len(f),5),np.int64);cand[len(sup):]=np.array(dv.loc[f.sample_id[len(sup):]].K_id.tolist())
    np.save(out/'cand.npy',cand)
    dump(out/'ready.json',{'n':len(f),'ns':len(sup),'nc':len(cal),'nclass':900})

def prepare_new(name):
    out=ROOT/'data'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest_frozen.json').exists():return
    seed=2026092706 if name=='cub' else 2026092707
    rng=np.random.default_rng(seed);allrows=[];cifar_data={}
    if name=='cub':
        archive=ROOT/'data/CUB_200_2011.tgz'
        if not archive.exists():raise FileNotFoundError('CUB archive is not downloaded')
        h=hashlib.md5()
        with archive.open('rb') as f:
            for part in iter(lambda:f.read(2**20),b''):h.update(part)
        if h.hexdigest()!='97eceeb196236b17998738112f37df78':raise ValueError('CUB official MD5 mismatch / download incomplete')
        if not (ROOT/'data/CUB_200_2011/images.txt').exists():
            with tarfile.open(archive) as t:t.extractall(ROOT/'data',filter='data')
        b=ROOT/'data/CUB_200_2011'
        rd=lambda fn:{int(s.split()[0]):' '.join(s.split()[1:]) for s in (b/fn).read_text().splitlines()}
        paths,labels,split=rd('images.txt'),rd('image_class_labels.txt'),rd('train_test_split.txt')
        names={i-1:n.split('.',1)[1].replace('_',' ') for i,n in rd('classes.txt').items()}
        for i,p in paths.items():allrows.append({'sample_id':f'cub:{i}','path':str(b/'images'/p),'original_class':int(labels[i])-1,'train':bool(int(split[i]))})
    else:
        from torchvision.datasets import CIFAR100
        b=Path('/home/omote/datasets/cifar100')
        images=out/'images';images.mkdir(exist_ok=True)
        for train in [True,False]:
            ds=CIFAR100(str(b),train=train,download=False);names=dict(enumerate(ds.classes))
            cifar_data[train]=ds.data
            for i,c in enumerate(ds.targets):
                sid=f'cifar:{"train" if train else "test"}:{i}';p=images/(sid.replace(':','_')+'.png')
                allrows.append({'sample_id':sid,'path':str(p),'original_class':c,'train':train,'source_index':i})
    nc=len(names);idclasses=sorted(rng.permutation(nc)[:nc//2].tolist());idmap={c:i for i,c in enumerate(idclasses)}
    allf=pd.DataFrame(allrows);pieces=[]
    for c in idclasses:
        f=allf[(allf.original_class==c)&allf.train]
        chosen=rng.permutation(f.index)[:20]
        if len(chosen)<20:raise ValueError(f'{name} class {c} has insufficient distinct training shots')
        for role,ix in [('support',chosen[:12]),('calib',chosen[12:16]),('id_threshold',chosen[16:20])]:
            ss=allf.loc[ix].copy();ss['role']=role;pieces.append(ss)
    e=allf[~allf.train].copy();e['role']='eval';pieces.append(e)
    f=pd.concat(pieces).copy();f['id_class']=f.original_class.map(idmap).fillna(-1).astype(int)
    f['class']=f.original_class.astype(str);f['is_ood']=~f.original_class.isin(idclasses);f['dataset']=name
    f['rank']=f.role.map({'support':0,'calib':1,'id_threshold':2,'eval':3})
    f=f.sort_values(['rank','original_class'],kind='stable').drop(columns='rank').reset_index(drop=True)
    if name=='cifar':
        for row in f.itertuples():
            if not Path(row.path).exists():Image.fromarray(cifar_data[row.train][row.source_index]).save(row.path)
    f=audit_images(f,out/'pixel_hashes.jsonl')
    # If fixed shots themselves duplicate, replace them from ID train rather than accept leakage.
    fixed=f[f.role.isin(['support','calib'])]
    if not fixed.pixel_sha256.is_unique:raise RuntimeError(f'{name}: duplicate support/calibration, requires a pre-evaluation split amendment')
    f.to_parquet(out/'images.parquet',index=False)
    info={'seed':seed,'nclass':nc//2,'ns':12*(nc//2),'nc':4*(nc//2),'n':len(f),
          'id_classes':idclasses,'id_names':[names[c] for c in idclasses],
          'ood_names_given_to_detector':False,'user_confirmed_development_unused':True,
          'pretraining_unseen_claim':False,'split_sha256':sha_file(out/'images.parquet'),
          'excluded_pixel_duplicates':int((~f.eligible).sum()),'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/'manifest_frozen.json',info)

def encode_new(name,model_name):
    from bridge import Encoder
    out=ROOT/'data'/name
    complete=(out/f'{model_name}.npy').exists() and (model_name!='CLIP' or (out/'cand.npy').exists())
    if complete:
        if all((out/f'{m}.npy').exists() for m in ['B14','L14','CLIP']) and (out/'cand.npy').exists():
            dump(out/'ready.json',json.loads((out/'manifest_frozen.json').read_text()))
        return
    f=pd.read_parquet(out/'images.parquet');enc=Encoder([model_name])
    chunks=[]
    for lo in range(0,len(f),64):
        result,times=enc.encode(f.path.iloc[lo:lo+64].tolist());chunks.append(result[model_name])
        if lo%640==0:print(json.dumps({'encode':name,'model':model_name,'done':lo+len(result[model_name]),'total':len(f)}),flush=True)
    tmp=out/f'{model_name}.{os.getpid()}.npy';np.save(tmp,np.concatenate(chunks));os.replace(tmp,out/f'{model_name}.npy')
    if model_name=='CLIP':
        info=json.loads((out/'manifest_frozen.json').read_text())
        pos=enc.text(info['id_names']);np.save(out/'pos.npy',pos)
        F=np.concatenate(chunks);cand=[]
        for lo in range(0,len(F),4096):cand.append(np.argsort(-(F[lo:lo+4096]@pos.T),axis=1)[:,:5])
        np.save(out/'cand.npy',np.concatenate(cand))
    if all((out/f'{m}.npy').exists() for m in ['B14','L14','CLIP']) and (out/'cand.npy').exists():
        dump(out/'ready.json',json.loads((out/'manifest_frozen.json').read_text()))

class Bank:
    def __init__(self,name):
        self.name=name;self.base=ROOT/'data'/name
        self.info=json.loads((self.base/'ready.json').read_text());self.frame=pd.read_parquet(self.base/'images.parquet')
        self.ns=self.info['ns'];self.nc=self.info['nc'];self.nclass=self.info['nclass']
        self.cand=np.load(self.base/'cand.npy',mmap_mode='r');self.features={};self.views={}
    def load(self,names):
        for name in names:
            self.features[name]=np.load(self.base/f'{name}.npy',mmap_mode='r')
            # Single-encoder controls use image prototypes only, including candidate selection.
            # The original dual condition retains the frozen CLIP-text candidate policy.
            image_only=len(names)==1
            cache=self.base/f'view_{name}{"_imageonly" if image_only else ""}.npz';F=self.features[name]
            if cache.exists():
                z=np.load(cache);v={k:z[k] for k in z.files}
                v['support']=F[:self.ns].reshape(self.nclass,12,-1);v['cal']=F[self.ns:self.ns+self.nc]
            else:
                candidates=self.cand
                if image_only:
                    mu=norm(np.asarray(F[:self.ns]).reshape(self.nclass,12,-1).mean(1))
                    candidates=np.concatenate([np.argsort(-(F[lo:lo+1024]@mu.T),axis=1)[:,:min(5,self.nclass)] for lo in range(0,len(F),1024)])
                v=static_view(F[:self.ns].reshape(self.nclass,12,-1),F[self.ns:self.ns+self.nc],F,
                              candidates[self.ns:self.ns+self.nc],candidates)
                import os
                tmp=str(cache)+f'.{os.getpid()}.npz';np.savez(tmp,**{k:v[k] for k in ['dcal','dallcal','d','p','pall','med','mad','mu','mt','st']});os.replace(tmp,cache)
            self.views[name]=v
        return self
