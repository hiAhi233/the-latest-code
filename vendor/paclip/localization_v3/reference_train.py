"""Train the same frozen-backbone BCE+Dice control on a fixed K-shot dataset.

Only source validation AP chooses a checkpoint; validation Dice fixes threshold.
No test image is encoded or used for selection in this stage.
"""
from pathlib import Path
import argparse
import collections
import json
import os
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import SliceSet,make_views,extract_views,sha,digest
from .model import FrozenEncoder,RecallDenseHead
from .metrics import operating_metrics,ranking_metrics,threshold_candidates
from .experiment import load_config,save_json,save_torch,cpu_state,ROOT


def pooled_seg_loss(logits,gt,roi):
    z=logits[roi];y=gt[roi].float()
    bce=F.binary_cross_entropy_with_logits(z,y,pos_weight=z.new_tensor(5.))
    p=z.sigmoid()
    dice=1-(2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)
    return .5*bce+.5*dice


@torch.no_grad()
def validate(head,items,device):
    head.eval();scores=[]
    for batch in [items[i:i+16] for i in range(0,len(items),16)]:
        f=torch.cat([r['features'] for r in batch]).to(device)
        scores.extend(head(f).sigmoid().cpu().numpy())
    masks=[r['gt'] for r in items];rois=[r['roi'] for r in items]
    return scores,masks,rois,ranking_metrics(scores,masks,rois)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    c=load_config(args.config);out=Path(c['output']);device=c['device']
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if any(x.strip() and x.strip()!=str(os.getpid()) for x in q.stdout.splitlines()):
        raise RuntimeError('GPU already occupied')
    data=SliceSet(c['manifest'],c['support'])
    support_ids={r['file'] for r in data.splits['train']}
    out.mkdir();started=time.time()
    sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.relative_to(ROOT).parts}
    from huggingface_hub import hf_hub_download
    weights=hf_hub_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224','open_clip_pytorch_model.bin',local_files_only=True)
    identity=digest({'config':c,'data':data.identity,'sources':sources,'pretrained_sha256':sha(weights)})
    save_json(out/'identity.json',{'digest':identity,'config':c,'data':data.identity,'source_sha256':sources,
                                 'role':f'new liver-specific K={len(support_ids)} reference, not a transferred thymoma head'})
    for rel in sources:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/rel).read_bytes())
    save_json(out/'status.json',{'status':'loading_frozen_encoder'})
    encoder=FrozenEncoder().to(device)
    train=[];validation=[]
    for split,destination in [('train',train),('val',validation)]:
        for row in data.splits[split]:
            arrays=data.read(row)
            images,_=make_views(*arrays,crop=c['crop_size'])
            destination.append({'features':extract_views(encoder,images[:1],device),
                                'gt':arrays[2],'roi':arrays[3],'row':row})
    del encoder;torch.cuda.empty_cache()
    features=torch.cat([r['features'] for r in train]).to(device)
    gt=torch.from_numpy(np.stack([r['gt'] for r in train])).to(device)
    roi=torch.from_numpy(np.stack([r['roi'] for r in train])).to(device)
    positive=[i for i,r in enumerate(train) if r['row']['label']]
    negative=[i for i,r in enumerate(train) if not r['row']['label']]
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    head=RecallDenseHead().to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=c['lr'],weight_decay=1e-5)
    rng=np.random.default_rng(c['seed']);access=collections.Counter();history=[]
    best_ap=-1.;best=None;best_step=None
    for step in range(1,c['steps']+1):
        head.train()
        pos=rng.choice(positive,4);neg=rng.choice(negative,4)
        a=torch.as_tensor(pos,device=device);b=torch.as_tensor(neg,device=device)
        loss=pooled_seg_loss(head(features[a]),gt[a],roi[a])+.25*pooled_seg_loss(head(features[b]),gt[b],roi[b])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reference loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        for i in np.r_[pos,neg]:access[train[int(i)]['row']['file']]+=1
        if step%c['validate_every']==0 or step==c['steps']:
            scores,masks,rois,m=validate(head,validation,device)
            ap=m['pixel_ap_roi_masked'];history.append({'step':step,'loss':float(loss.detach()),'validation_ap':ap})
            if ap>best_ap:best_ap=ap;best=cpu_state(head);best_step=step
            save_json(out/'status.json',{'status':'training','completed_steps':step})
    head.load_state_dict(best)
    scores,masks,rois,m=validate(head,validation,device)
    candidates=[operating_metrics(scores,masks,rois,t) for t in threshold_candidates(scores,masks,rois)]
    safe=[v for v in candidates if v['normal_fp_area'] <= c['max_normal_fp']+1e-12]
    if not safe:raise ValueError('No validation threshold satisfies the normal FP budget')
    choice=max(safe,key=lambda v:v['dice_micro'])
    state={'config':'M1_liver_fresh_reference','fusion':'indep','pseudo':None,'adapter_spec':None,
           'state':best,'steps':c['steps'],'best_step':best_step,'seed':c['seed'],
           'pixel_threshold':choice['threshold'],'image_threshold':None,'identity':identity,
           'max_normal_fp':c['max_normal_fp'],'support_count':len(support_ids),
           'source_support_manifest_sha256':sha(c['support']),'source_data_manifest_sha256':sha(c['manifest']),
           'method':'same spatial head, frozen encoder, pooled weighted BCE+Dice, 4normal4abnormal, no zoom/flip'}
    (out/'sr2_states').mkdir();save_torch(out/'sr2_states/M1.pt',state)
    save_json(out/'validation.json',{**m,**choice})
    save_json(out/'training_history.json',history);save_json(out/'training_access.json',dict(access))
    assert set(access)==support_ids and sum(access.values())==c['steps']*8
    save_json(out/'completion.json',{'status':'complete','steps':c['steps'],'selected_step':best_step,
                                    'checkpoint_sha256':sha(out/'sr2_states/M1.pt'),'elapsed_seconds':time.time()-started})
    save_json(out/'status.json',{'status':'complete'})
    print(json.dumps({'status':'complete','selected_step':best_step,'validation':choice},ensure_ascii=False),flush=True)
