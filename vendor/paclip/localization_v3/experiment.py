"""Auditable K-shot development: fit on support, select on val, evaluate test once."""
from pathlib import Path
import argparse
import collections
import json
import os
import shutil
import subprocess
import time
import numpy as np
import torch
from PIL import Image
from .data import SliceSet, make_views, extract_views, sha, digest
from .model import FrozenEncoder, RecallDenseHead, combine_views, view_boxes
from .loss import recall_loss
from .metrics import operating_metrics, ranking_metrics, select_validation, selection_key, standard_ranking_metrics
from .normal_gate import build_memory, distance_maps, fit_normal_cdf, gate_scores
from .metrics import threshold_candidates

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.generic): return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v): return None
        return v
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def save_torch(path, value):
    tmp = Path(str(path)+'.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def load_config(path):
    path=Path(path).resolve()
    c=json.loads(path.read_text(encoding='utf-8'))
    allowed={'manifest','support','reference_checkpoint','output','steps','seed','lr','batch_size',
             'validate_every','crop_size','zoom_weights','recall_weight','device','base_checkpoint','normal_gate_weights','task','max_normal_fp'}
    if set(c)-allowed: raise ValueError(f'Unknown config keys: {set(c)-allowed}')
    for k in ['manifest','support','reference_checkpoint','output']:
        c[k]=str((path.parent/c[k]).resolve())
    if c.get('base_checkpoint'):
        c['base_checkpoint']=str((path.parent/c['base_checkpoint']).resolve())
    if c['steps']<1 or c['batch_size']<2 or c['batch_size']%2:
        raise ValueError('Positive steps and even batch required')
    if c['validate_every']<1 or c['lr']<=0 or c['recall_weight']<0:
        raise ValueError('Invalid optimizer/validation configuration')
    if any(a<0 or a>1 for a in c['zoom_weights']): raise ValueError('Invalid fusion weights')
    c.setdefault('max_normal_fp',1.)
    if not 0 <= c['max_normal_fp'] <= 1: raise ValueError('Invalid normal FP budget')
    if c.get('task','brain_mri') not in ('brain_mri','thymoma','liver_ct'):
        raise ValueError('Unsupported task')
    view_boxes(crop=c['crop_size'])
    return c


def build_cache(encoder,data,device,crop):
    train_features=[];targets=[];metadata=[]
    for row in data.splits['train']:
        arrays=data.read(row,native=True)
        for flip in (False,True):
            images,t=make_views(*arrays,flip=flip,crop=crop)
            features=extract_views(encoder,images,device)
            for j,(f,target) in enumerate(zip(features,t)):
                if not target['roi'].any(): continue
                train_features.append(f);targets.append(target)
                metadata.append({'sample':row['file'],'view':j,'flip':flip,
                                 'positive':bool(target['hard'].any())})
    validation=[]
    for row in data.splits['val']:
        arrays=data.read(row)
        images,_=make_views(*arrays,crop=crop)
        validation.append({'features':extract_views(encoder,images,device),
                           'gt':arrays[2],'roi':arrays[3],'row':row})
    return {'train_features':torch.stack(train_features),
            'targets':{k:torch.stack([t[k] for t in targets]) for k in targets[0]},
            'metadata':metadata,'validation':validation}


@torch.no_grad()
def predict_cached(head,validation,device,crop):
    head.eval();global_maps=[];zoom_maps=[]
    for item in validation:
        p=head(item['features'].to(device)).sigmoid()
        a,b=combine_views(p,view_boxes(crop=crop))
        global_maps.append(a.cpu().numpy());zoom_maps.append(b.cpu().numpy())
    return np.stack(global_maps),np.stack(zoom_maps)


def training(encoder,reference,data,c,out,identity,resume):
    device=c['device'];cache_path=out/'support_validation_cache.pt'
    if cache_path.exists():
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        if cache['identity']!=identity: raise ValueError('Feature cache identity mismatch')
    else:
        cache=build_cache(encoder,data,device,c['crop_size']);cache['identity']=identity
        save_torch(cache_path,cache)
    validation=cache['validation']
    masks=[i['gt'] for i in validation];rois=[i['roi'] for i in validation]
    rows=[i['row'] for i in validation]
    ref_maps,_=predict_cached(reference,validation,device,c['crop_size'])
    ref_threshold=float(torch.load(c['reference_checkpoint'],map_location='cpu',weights_only=False)['pixel_threshold'])
    ref_metrics=operating_metrics(ref_maps,masks,rois,ref_threshold)
    if ref_metrics['small_lesion_recall'] is None: raise ValueError('No validation small lesions')
    save_json(out/'reference_validation.json',ref_metrics)
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    head=RecallDenseHead().to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['lr'],weight_decay=1e-5)
    rng=np.random.default_rng(c['seed'])
    access=collections.Counter();history=[];start=0;best_key=None
    if resume:
        state=torch.load(out/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        if state['identity']!=identity: raise ValueError('Resume identity mismatch')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['step'];access.update(state['access']);history=state['history']
        if (out/'checkpoint_selected.pt').exists():
            best_key=tuple(torch.load(out/'checkpoint_selected.pt',map_location='cpu',weights_only=False)['selection_key'])
    features=cache['train_features'].to(device)
    targets={k:v.to(device) for k,v in cache['targets'].items()}
    positive=[i for i,m in enumerate(cache['metadata']) if m['positive']]
    negative=[i for i,m in enumerate(cache['metadata']) if not m['positive']]
    if not positive or not negative: raise ValueError('No usable training class')
    half=c['batch_size']//2
    for step in range(start+1,c['steps']+1):
        head.train()
        idx=np.r_[rng.choice(positive,half),rng.choice(negative,half)]
        ix=torch.tensor(idx,device=device)
        loss=recall_loss(head(features[ix]),{k:v[ix] for k,v in targets.items()},recall_weight=c['recall_weight'])
        if not torch.isfinite(loss['total']): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss['total'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        for i in idx: access[cache['metadata'][int(i)]['sample']]+=1
        if step%c['validate_every']==0 or step==c['steps']:
            global_maps,zoom_maps=predict_cached(head,validation,device,c['crop_size'])
            choice,profiles=select_validation(global_maps,zoom_maps,masks,rois,ref_metrics,rows,c['zoom_weights'],c['max_normal_fp'])
            key=selection_key(choice['metrics'],ref_metrics,c['max_normal_fp'])
            history.append({'step':step,'loss':{k:float(v.detach()) for k,v in loss.items()},**choice})
            if best_key is None or key>best_key:
                best_key=key
                save_torch(out/'checkpoint_selected.pt',{'format':'recall_dense_v3','identity':identity,
                    'head':cpu_state(head),'step':step,'choice':choice,'selection_key':key,'config':c,
                    'reference_validation':ref_metrics,'trainable_encoder_parameters':0})
                save_json(out/'selected_validation_profiles.json',profiles)
            save_torch(out/'checkpoint_last.pt',{'identity':identity,'head':cpu_state(head),
                'optimizer':optimizer.state_dict(),'step':step,'rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'access':dict(access),'history':history})
            save_json(out/'training_history.json',history)
            save_json(out/'status.json',{'status':'training','completed_steps':step,'selected_validation':choice})
            print(f'[validation] step={step} dice={choice["metrics"]["dice_micro"]:.4f} small={choice["metrics"]["small_lesion_recall"]:.4f} guard={choice["guard_passed"]}',flush=True)
    save_json(out/'training_access.json',dict(access))
    if set(access)!={r['file'] for r in data.splits['train']}:
        raise AssertionError('Support access mismatch')
    return torch.load(out/'checkpoint_selected.pt',map_location='cpu',weights_only=False)


@torch.no_grad()
def calibrate_normal(reference,data,c,out,identity):
    """Reuse completed candidate training; only validation chooses the new gate."""
    base=Path(c['base_checkpoint'])
    original=json.loads((base.parent/'identity.json').read_text(encoding='utf-8'))
    if original['data']!=data.identity or original['config']['crop_size']!=c['crop_size']:
        raise ValueError('Base cache has different data/crop preprocessing')
    for rel in ('localization_v3/data.py','localization_v3/model.py','models/paclip.py','config.py'):
        recorded=original['source_sha256'].get(rel,original['source_sha256'].get(str(Path(rel))))
        if recorded!=sha(ROOT/rel):
            raise ValueError('Frozen extractor/preprocessing changed since base cache')
    cache=torch.load(base.parent/'support_validation_cache.pt',map_location='cpu',weights_only=False)
    selected=torch.load(base,map_location='cpu',weights_only=False)
    if cache['identity']!=original['digest'] or selected['identity']!=original['digest']:
        raise ValueError('Base checkpoint/cache identity mismatch')
    validation=cache['validation'];masks=[x['gt'] for x in validation];rois=[x['roi'] for x in validation]
    rows=[x['row'] for x in validation]
    head=RecallDenseHead().to(c['device']);head.load_state_dict(selected['head']);head.eval()
    global_maps,zoom_maps=predict_cached(head,validation,c['device'],c['crop_size'])
    ref_maps,_=predict_cached(reference,validation,c['device'],c['crop_size'])
    threshold=float(torch.load(c['reference_checkpoint'],map_location='cpu',weights_only=False)['pixel_threshold'])
    reference_metrics=operating_metrics(ref_maps,masks,rois,threshold)
    memory=build_memory(cache,{r['file'] for r in data.splits['train'] if r['label']==0})
    dg=[];dz=[]
    for item in validation:
        d=distance_maps(item['features'].to(c['device']),memory)
        a,b=combine_views(d,view_boxes(crop=c['crop_size']))
        dg.append(a.cpu().numpy());dz.append(b.cpu().numpy())
    dg=np.stack(dg);dz=np.stack(dz)
    best=None;profiles=[]
    for alpha in c['zoom_weights']:
        scores=(1-alpha)*global_maps+alpha*zoom_maps
        distances=(1-alpha)*dg+alpha*dz
        cdf=fit_normal_cdf(distances,masks,rois,rows)
        for weight in c['normal_gate_weights']:
            candidate=gate_scores(scores,distances,cdf,weight)
            for t in threshold_candidates(candidate,masks,rois):
                m=operating_metrics(candidate,masks,rois,t);key=selection_key(m,reference_metrics,c['max_normal_fp'])
                choice={'alpha':alpha,'normal_gate_weight':weight,'metrics':m,
                        'guard_passed':bool(key[0]),'selection_score':key[1]}
                profiles.append(choice)
                if best is None or key>best[0]: best=(key,choice,cdf)
    selected.update({'identity':identity,'base_checkpoint_sha256':sha(base),'config':c,
                     'selection_key':best[0],'choice':best[1], 'normal_memory':memory,
                     'normal_cdf':best[2],'reference_validation':reference_metrics})
    save_torch(out/'checkpoint_selected.pt',selected)
    save_json(out/'selected_validation_profiles.json',profiles)
    save_json(out/'reference_validation.json',reference_metrics)
    shutil.copy2(base.parent/'training_access.json',out/'training_access.json')
    save_json(out/'calibration.json',{'base_checkpoint':str(base),'base_checkpoint_sha256':sha(base),
        'base_selected_step':selected['step'],'normal_training_ids':sorted(r['file'] for r in data.splits['train'] if r['label']==0),
        'choice':best[1],'calibration_source':'normal_validation_only','test_access_for_selection':False})
    return selected


def make_gallery(out, rows, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positives=sorted([r for r in rows if r['gt_pixels']],key=lambda r:r['gt_pixels'])
    negatives=[r for r in rows if not r['gt_pixels']]
    # Fixed GT strata; no success/failure-based selection.
    selected=[positives[int(q*(len(positives)-1))] for q in (0,.05,.2,.4,.6,.8,.95)]
    if negatives: selected.append(negatives[0])
    fig,axes=plt.subplots(3,len(selected),figsize=(3*len(selected),9),squeeze=False)
    for j,row in enumerate(selected):
        record=next(r for r in data.splits['test'] if r['file']==row['file'])
        gray,_,gt,roi=data.read(record)
        d=np.load(out/'test_predictions'/row['file'])
        for k in range(3):
            ax=axes[k,j];ax.imshow(gray,cmap='gray',vmin=0,vmax=1);ax.axis('off')
            if gt.any(): ax.contour(gt,levels=[.5],colors=['lime'],linewidths=.7)
        axes[0,j].set_title(f'{Path(row["file"]).stem}\nGT {int(gt.sum())} px',fontsize=8)
        for k,key in [(1,'reference'),(2,'candidate')]:
            score=d[key+'_score'];pred=d[key+'_prediction']
            axes[k,j].imshow(np.ma.masked_where(~roi,score),cmap='turbo',vmin=0,vmax=1,alpha=.35)
            if pred.any() and not pred.all(): axes[k,j].contour(pred,levels=[.5],colors=['red'],linewidths=.7)
            axes[k,j].set_title(f'{key}: Dice {row[key+"_dice"]:.3f}',fontsize=9)
    fig.tight_layout();fig.savefig(out/'matched_eight.png',dpi=130);plt.close(fig)


@torch.no_grad()
def evaluate(encoder,reference,selected,data,c,out):
    head=RecallDenseHead().to(c['device']);head.load_state_dict(selected['head']);head.eval()
    choice=selected['choice'];alpha=choice['alpha'];threshold=choice['metrics']['threshold']
    ref_threshold=float(torch.load(c['reference_checkpoint'],map_location='cpu',weights_only=False)['pixel_threshold'])
    all_scores={'reference':[],'candidate':[]};masks=[];rois=[];rows=[]
    pred_dir=out/'test_predictions';pred_dir.mkdir(exist_ok=True)
    for number,row in enumerate(data.splits['test'],1):
        gray,soft,gt,roi=data.read(row)
        images,_=make_views(gray,soft,gt,roi,crop=c['crop_size'])
        features=extract_views(encoder,images,c['device']).to(c['device'])
        base=reference(features[:1]).sigmoid()[0].cpu().numpy()
        p=head(features).sigmoid()
        global_map,zoom_map=combine_views(p,view_boxes(crop=c['crop_size']))
        score=((1-alpha)*global_map+alpha*zoom_map).cpu().numpy()
        if choice.get('normal_gate_weight',0)>0:
            d=distance_maps(features,selected['normal_memory'])
            dg,dz=combine_views(d,view_boxes(crop=c['crop_size']))
            distance=((1-alpha)*dg+alpha*dz).cpu().numpy()
            score=gate_scores(score,distance,selected['normal_cdf'],choice['normal_gate_weight'])
        reference_pred=(base>=ref_threshold)&roi;candidate_pred=(score>=threshold)&roi
        np.savez_compressed(pred_dir/row['file'],candidate_score=score,reference_score=base,
                            candidate_prediction=candidate_pred,reference_prediction=reference_pred,gt=gt,roi=roi)
        record={'file':row['file'],'gt_pixels':int(gt.sum())}
        for name,pred,s in [('reference',reference_pred,base),('candidate',candidate_pred,score)]:
            den=int(pred.sum())+int(gt.sum())
            record[name+'_dice']=2*int((pred & gt).sum())/den if den else 1.
            all_scores[name].append(s)
        rows.append(record);masks.append(gt);rois.append(roi)
        if number%250==0:
            save_json(out/'status.json',{'status':'test_inference','done':number,'total':len(data.splits['test'])})
    results={}
    for name,t in [('reference',ref_threshold),('candidate',threshold)]:
        results[name]={**operating_metrics(all_scores[name],masks,rois,t),
                       **ranking_metrics(all_scores[name],masks,rois),
                       **standard_ranking_metrics(all_scores[name],masks,rois)}
    delta={k:results['candidate'][k]-results['reference'][k] for k in
           ['image_auroc','pixel_auroc','pixel_ap_roi_masked','dice_micro','lesion_recall','small_lesion_recall','normal_fp_area']}
    results['delta']=delta
    results['all_three_test_improved']=delta['dice_micro']>0 and delta['small_lesion_recall']>0 and delta['normal_fp_area']<0
    results['validation_guard_passed']=choice['guard_passed']
    results['selection_note']='Selected on validation only; test evaluated after checkpoint/alpha/threshold frozen.'
    results['patient_disjoint_verified']=data.patient_verified
    results['metric_protocol']={'image_auroc':'maximum of full final continuous anomaly map',
        'pixel_auroc':'flatten all final continuous maps and all GT pixels; no extra ROI suppression',
        'roi_metrics':'auxiliary; not interchangeable with primary full-grid metrics'}
    save_json(out/'metrics.json',results);save_json(out/'test_per_slice.json',rows)
    make_gallery(out,rows,data)
    task_label={'thymoma':'胸腺瘤CT','liver_ct':'BMAD肝脏CT'}.get(c.get('task'),'脑MRI')
    scope=('本轮使用对方重建的胸腺瘤切片划分：训练池1548张，验证220张，测试466张；仅五张支持图参与梯度训练。'
           '正常图为病灶患者的无病灶切片，没有NLST独立正常患者；不是原585张测试协议。'
           '患者编号前缀跨划分无交叉，真实患者别名身份未独立认证。'
           if c.get('task')=='thymoma' else
           '原MRI患者身份独立性未认证，五张支持图不等于五个独立患者；结果属于单种子探索。')
    if c.get('task')=='liver_ct':
        scope=('本轮为BMAD数据上重新按源体积编号分组的小批监督试验，不是官方正常图训练协议。'
               '原官方valid/test共享LiTS体积编号；本轮重新固定三组，源体积编号跨划分无交叉。'
               f'训练{len(data.splits["train"])}张支持图，验证{len(data.splits["val"])}张、测试{len(data.splits["test"])}张。'
               f'验证选模的正常ROI平均误报面积上限为{100*c["max_normal_fp"]:.2f}%，测试集不保证满足同一上限。'
               '输入已由BMAD按肝脏标注去除肝外背景，不代表全腹部自动定位；真实患者别名未独立认证。')
    lines=[f'# {task_label} K={len(data.splits["train"])}：原始代码上新增定位分支','',
           '固定支持图训练；验证集选择权重、缩放视图融合和阈值；测试集不参与选择。',
           f'完成训练{c["steps"]}步，选用第{selected["step"]}步，局部视图权重={alpha}，正常参照权重={choice.get("normal_gate_weight",0)}。',
           f'验证集三项保护通过：{choice["guard_passed"]}；测试集三项同时严格改善：{results["all_three_test_improved"]}。','',
           '| 指标 | 已有冻结主干密集头 | 新定位分支 | 差值（百分点） |','|---|---:|---:|---:|']
    for key,title in [('image_auroc','Image AUROC（全图最大异常分数）'),
                      ('pixel_auroc','Pixel AUROC（全224网格，原始连续分数）'),
                      ('pixel_ap_roi_masked','ROI限制Pixel AP'),('dice_micro','Dice micro'),
                      ('lesion_recall','严格病灶召回'),('small_lesion_recall','严格小病灶召回'),('normal_fp_area','正常ROI误报面积')]:
        lines.append(f'| {title} | {results["reference"][key]*100:.4f}% | {results["candidate"][key]*100:.4f}% | {delta[key]*100:+.4f} |')
    lines+=['','本次未使用旧适配器或旧密集头权重初始化候选模型。已有密集头权重只作为固定对照。',
            scope,'本轮仅seed=0，指标差异不等于多种子稳定收益。',
            '局部放大采用固定重叠窗口，不用测试GT选位置。所有测试GT（包括ROI外病灶）均保留在指标分母。','',
            '![同病例定位](matched_eight.png)','',
            '完整逐图预测位于test_predictions，训练取样记录见training_access.json。']
    (out/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return results


@torch.no_grad()
def predict_image(args):
    selected=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if selected.get('format')!='recall_dense_v3': raise ValueError('Not a v3 checkpoint')
    c=selected['config'];device=c['device']
    encoder=FrozenEncoder().to(device);head=RecallDenseHead().to(device)
    head.load_state_dict(selected['head']);head.eval()
    with Image.open(args.predict) as im:
        gray=np.asarray(im.convert('L').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
    if args.roi_mask:
        with Image.open(args.roi_mask) as im:
            roi=np.asarray(im.convert('L').resize((224,224),Image.Resampling.NEAREST))>0
    elif c.get('task') in ('thymoma','liver_ct'):
        raise ValueError('CT PNG inference requires --roi-mask with the same body ROI protocol')
    else:
        roi=gray>20/255
    if not roi.any(): raise ValueError('Empty inference ROI')
    empty=np.zeros_like(gray)
    images,_=make_views(gray,empty,empty,roi,crop=c['crop_size'])
    features=extract_views(encoder,images,device).to(device)
    p=head(features).sigmoid()
    a,b=combine_views(p,view_boxes(crop=c['crop_size']));choice=selected['choice']
    score=((1-choice['alpha'])*a+choice['alpha']*b).cpu().numpy()
    if choice.get('normal_gate_weight',0)>0:
        dg,dz=combine_views(distance_maps(features,selected['normal_memory']),view_boxes(crop=c['crop_size']))
        d=((1-choice['alpha'])*dg+choice['alpha']*dz).cpu().numpy()
        score=gate_scores(score,d,selected['normal_cdf'],choice['normal_gate_weight'])
    pred=(score>=choice['metrics']['threshold'])&roi
    dest=Path(args.prediction_output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,score=score,prediction=pred,roi=roi)
    print(str(dest),flush=True)


def main():
    parser=argparse.ArgumentParser(description='Frozen BiomedCLIP, small-lesion supervision and zoom localization')
    parser.add_argument('--config',default=str(ROOT/'configs/mri_k5.json'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--predict');parser.add_argument('--checkpoint')
    parser.add_argument('--roi-mask',help='Body ROI PNG required for CT single-image inference')
    parser.add_argument('--prediction-output',default='prediction.npz')
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark=False
    if args.predict:
        if not args.checkpoint: parser.error('--predict requires --checkpoint')
        return predict_image(args)
    c=load_config(args.config);out=Path(c['output'])
    if c['device'].startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if c['device'].startswith('cuda'):
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        others=[x.strip() for x in q.stdout.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
        if others: raise RuntimeError('Another GPU process is active: '+','.join(others))
    if out.exists() and not (args.resume or args.evaluate_only): raise FileExistsError('Use a fresh output or --resume')
    out.mkdir(parents=True,exist_ok=True);started=time.time()
    save_json(out/'status.json',{'status':'preflight'})
    data=SliceSet(c['manifest'],c['support'])
    source_files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.relative_to(ROOT).parts}
    from huggingface_hub import hf_hub_download
    weights=hf_hub_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224','open_clip_pytorch_model.bin',local_files_only=True)
    identity=digest({'config':c,'data':data.identity,'source':source_files,'pretrained_sha256':sha(weights),
                     'reference_sha256':sha(c['reference_checkpoint']),
                     'base_checkpoint_sha256':sha(c['base_checkpoint']) if c.get('base_checkpoint') else None})
    if (args.resume or args.evaluate_only) and (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text(encoding='utf-8'))['digest']!=identity:
            raise ValueError('Experiment identity changed; use a fresh output directory')
    save_json(out/'identity.json',{'digest':identity,'config':c,'source_sha256':source_files,
        'data':data.identity,'patient_disjoint_verified':data.patient_verified,
        'package_root':str(ROOT),'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda}})
    for rel in source_files:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(out/'config.json',c)
    save_json(out/'status.json',{'status':'loading_frozen_encoder'})
    encoder=FrozenEncoder().to(c['device']);encoder.eval()
    if any(p.requires_grad for p in encoder.parameters()): raise AssertionError('Encoder not frozen')
    ref_blob=torch.load(c['reference_checkpoint'],map_location='cpu',weights_only=False)
    if ref_blob.get('adapter_spec') is not None or ref_blob.get('fusion')!='indep':
        raise ValueError('Reference must be independent-projection head without adapters')
    reference=RecallDenseHead().to(c['device']);reference.load_state_dict(ref_blob['state'],strict=True);reference.eval()
    if args.evaluate_only:
        selected=torch.load(out/'checkpoint_selected.pt',map_location='cpu',weights_only=False)
        if selected['identity']!=identity: raise ValueError('Checkpoint identity mismatch')
    elif c.get('base_checkpoint'):
        selected=calibrate_normal(reference,data,c,out,identity)
    else:
        selected=training(encoder,reference,data,c,out,identity,args.resume)
    save_json(out/'status.json',{'status':'final_test','selected_step':selected['step']})
    results=evaluate(encoder,reference,selected,data,c,out)
    save_json(out/'completion.json',{'status':'complete','identity':identity,'steps':c['steps'],
        'selected_step':selected['step'],'test_count':len(data.splits['test']),
        'checkpoint_sha256':sha(out/'checkpoint_selected.pt'),'metrics_sha256':sha(out/'metrics.json'),
        'elapsed_seconds':time.time()-started,'all_three_test_improved':results['all_three_test_improved']})
    save_json(out/'status.json',{'status':'complete'})
    print(json.dumps(results,ensure_ascii=False),flush=True)
