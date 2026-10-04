import hashlib,json,subprocess,sys
from pathlib import Path
import pandas as pd,torch
H=Path(__file__).resolve().parent;B=Path('/home/omote/vins_gonogo_20260925')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    out={'source_correspondence':{},'intra_split_duplicates':{},'models':[], 'features':[]}
    pr=json.loads((B/'r5/phase3/prereg_phase3.json').read_text())
    for path,expected in pr['code'].items():
        p=H/'vendor'/path
        if p.exists():out['source_correspondence'][path]={'registered':expected,'snapshot':sha(p),'matches':sha(p)==expected}
    out['documented_original_deviations']=json.loads((B/'r5/phase3/deviations.json').read_text())
    for dev,root in [('dev1',B),('dev2',B/'dev2')]:
        d=pd.read_parquet(root/'splits/samples.parquet')
        ev=d[d.split.isin(['id_dev','near_dev','far_dev'])]
        out['intra_split_duplicates'][dev]={'eval_duplicate_hash_rows':int(ev.sha256.duplicated().sum()),
             'cross_ID_OOD_hash_overlap':len(set(ev[ev.split=='id_dev'].sha256)&set(ev[ev.split!='id_dev'].sha256)),
             'duplicate_detail':ev[ev.sha256.duplicated(keep=False)][['sample_id','split','sha256']].to_dict('records')}
    paths=[Path('/home/omote/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'),
           Path('/home/omote/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'),
           Path('/home/omote/ood_large_best_20260923/tins_20260925/weights/ViT-B-16.pt')]
    for p in paths:out['models'].append({'path':str(p),'exists':p.exists(),'sha256':sha(p) if p.exists() else None})
    for name in ['dino.pt','dino_vitl14.pt','clip.pt']:
        z=torch.load(B/'features'/name,map_location='cpu');out['features'].append({'name':name,'meta':z.get('meta'),
                                    'shape':list(z['features'].shape),'dtype':str(z['features'].dtype),
                                    'unique_ids':len(set(z['sample_id'])),'count':len(z['sample_id'])})
    for p in [B/'tins/eval_tins_w_init.py',H/'vendor/vins/features.py',H/'vendor/scripts/r5_features.py']:
        out.setdefault('extractor_and_base_code',[]).append({'path':str(p),'sha256':sha(p)})
    out['resources']={'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True),
                      'CPU_threads':subprocess.check_output(['nproc'],text=True).strip(),
                      'memory':subprocess.check_output(['free','-b'],text=True),'disk':subprocess.check_output(['df','-B1',str(H)],text=True)}
    out['packages']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    (H/'provenance/audit_details.json').write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps({'code_mismatch':[k for k,v in out['source_correspondence'].items() if not v['matches']],
                      'duplicate_summary':{k:{x:y for x,y in v.items() if x!='duplicate_detail'} for k,v in out['intra_split_duplicates'].items()}}),flush=True)


if __name__=='__main__':main()
