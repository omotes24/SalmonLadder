"""Frozen model adapters and an online TINS state using the pinned upstream functions."""
import sys,time,json,copy,fcntl,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data import ROOT,OLD,VENDOR,loadpt,dump
sys.path.insert(0,str(VENDOR))
from vins import config as C
C.TINS_DIR=VENDOR/'tins'
from vins.tins_dev import import_tins,make_args,load_clip,get_logger,setup_to_device

def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()

class Encoder:
    def __init__(self,names):
        self.names=names;self.models={};self.transforms={}
        from vins.features import dino_transform
        for name in names:
            if name=='CLIP':
                self.t=import_tins(tins_dir=VENDOR/'tins')
                self.args=make_args(self.t,ROOT/'cache/encoder','controls_encoder')
                m,pre=load_clip(self.t,self.args)
            else:
                tag={'B14':'vitb14','L14':'vitl14'}[name]
                m=torch.hub.load(str(C.DINO_HUB),'dinov2_'+tag,source='local',pretrained=False)
                m.load_state_dict(loadpt(C.HOME/'.cache/torch/hub/checkpoints'/f'dinov2_{tag}_pretrain.pth'))
                m=m.eval().cuda();pre=dino_transform()
            self.models[name]=m;self.transforms[name]=pre

    @torch.no_grad()
    def encode(self,paths):
        # Loading/decoding/transforms are included in these timings.
        result={};times={}
        tick=time.perf_counter()
        images=[]
        for p in paths:
            with Image.open(p) as im:images.append(im.convert('RGB'))
        times['image_decode_s']=time.perf_counter()-tick
        for name,m in self.models.items():
            tick=time.perf_counter();chunks=[]
            for lo in range(0,len(images),32):
                x=torch.stack([self.transforms[name](im) for im in images[lo:lo+32]]).cuda()
                f=m.encode_image(x).float() if name=='CLIP' else m.forward_features(x)['x_norm_clstoken'].float()
                f=f/f.norm(dim=-1,keepdim=True);chunks.append(f.cpu().numpy())
            result[name]=np.concatenate(chunks);sync();times[name+'_s']=time.perf_counter()-tick
        return result,times

    @torch.no_grad()
    def text(self,labels):
        m=self.models['CLIP'];t=self.t
        f=t.encode_texts(m,[self.args.pos_prompt.format(x) for x in labels],batch_size=1000,device='cuda',desc='ID names')
        f=f.float();f=f/f.norm(dim=-1,keepdim=True);return f.cpu().numpy()

class Tins:
    def __init__(self,bank,encoder=None):
        self.encoder=encoder or Encoder(['CLIP']);self.t=self.encoder.t;self.model=self.encoder.models['CLIP']
        self.args=make_args(self.t,ROOT/'cache/tins'/bank.name,'controls_'+bank.name)
        self.args.batch_size=256
        self.t.setup_seed(0)
        setup_path=ROOT/'cache'/f'tins_setup_{bank.name}.pt';setup_path.parent.mkdir(parents=True,exist_ok=True)
        lock=setup_path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        if setup_path.exists():z=loadpt(setup_path)
        else:
            if bank.name=='openood':labels=[str(x) for x in np.load(OLD/'tins/data/ImageNet/imagenet_class_clean.npy')]
            elif bank.name=='dev':labels=[x['clean_name'] for x in json.loads((OLD/'splits/id_classes.json').read_text())]
            else:labels=bank.info['id_names']
            # Only ID labels enter the detector; OOD class names are never passed here.
            f=np.load(bank.base/'CLIP.npy',mmap_mode='r');sup=f[:bank.ns].reshape(bank.nclass,12,-1);cal=f[bank.ns:bank.ns+bank.nc].reshape(bank.nclass,4,-1)
            proto=torch.as_tensor(np.concatenate([sup,cal],1).mean(1),device='cuda');proto/=proto.norm(dim=-1,keepdim=True)
            pos=self.t.encode_texts(self.model,[self.args.pos_prompt.format(x) for x in labels],batch_size=1000,device='cuda',desc='ID names').to('cuda')
            log=get_logger(ROOT/'logs'/f'tins_setup_{bank.name}.log')
            neg,_,words,_=self.t.load_or_build_negative_bank(args=self.args,model=self.model,positive_labels=labels,positive_features=pos,class_prototypes=proto.cpu(),log=log)
            init=self.t.build_inversion_init_candidates(self.args,self.model,words,proto,'cuda',log)
            z={'positive_features':pos.cpu(),'negative_features':neg.cpu(),'class_prototypes':proto.cpu(),'base_sim':(pos*proto).sum(1).cpu(),
               'init_candidates':{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in init.items()}}
            tmp=setup_path.with_name(setup_path.name+f'.{os.getpid()}.tmp');torch.save(z,tmp);os.replace(tmp,setup_path)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        self.z=setup_to_device(z);self.reset()

    def reset(self):
        d=self.z['positive_features'].shape[1]
        self.bf=torch.zeros((0,d),dtype=torch.float16);self.bs=torch.zeros(0)
        self.uf=self.bf.clone();self.us=self.bs.clone()
        self.fixed=torch.cat([self.z['positive_features'],self.z['negative_features']])

    def step(self,features):
        t=self.t;a=self.args;z=self.z;q=torch.as_tensor(features,device='cuda',dtype=torch.float32)
        base=z['negative_features']
        neg=torch.cat([base,self.bf[:a.extra_text_length].float().cuda()]) if len(self.bf) else base
        def score(ng):
            return t.compute_grouped_positive_score(image_features=q,positive_features=z['positive_features'],negative_features=ng,
                    logit_scale=float(self.model.logit_scale.exp().detach().cpu()),group_num=a.group_num,random_permute=a.random_permute)
        with torch.no_grad():s=score(neg)
        updated,self.bf,self.bs,self.uf,self.us=t.maybe_expand_dynamic_bank(args=a,model=self.model,image_features=q,current_scores=s,
                class_prototypes=z['class_prototypes'],base_sim=z['base_sim'],fixed_text_bank=self.fixed,
                bank_features=self.bf,bank_scores=self.bs,placeholder_token_id=None,init_candidates=z['init_candidates'],
                buffer_features=self.uf,buffer_scores=self.us)
        if updated is not None:
            with torch.no_grad():s=score(updated[len(z['positive_features']):])
        return s.detach().cpu().numpy()
