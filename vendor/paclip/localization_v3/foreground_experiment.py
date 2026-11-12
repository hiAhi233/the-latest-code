"""Independent foreground-head training and full-map evaluation, no adapters."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
import torch
from PIL import Image
from .data import SliceSet,sha,digest,make_views,extract_views
from .model import FrozenEncoder,combine_views,view_boxes
from .foreground import ForegroundSeparatedModel,foreground_loss
from .background_experiment import choose
from .experiment import ROOT,save_json,save_torch,cpu_state
from .metrics import operating_metrics,standard_ranking_metrics
from .streaming_metrics import OperatingAccumulator,rank_npz


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fuse_views(maps,c):
    a,b=combine_views(maps,view_boxes(crop=c['crop_size']))
    return (1-c['alpha'])*a+c['alpha']*b


@torch.no_grad()
def validation_maps(model,items,c):
    if any(i['row']['split']!='val' for i in items):raise ValueError('Validation cache only')
    model.eval();base=[];final=[]
    for item in items:
        result=model(item['features'].to(c['device']))
        base.append(fuse_views(result['anomaly_probability'],c).cpu().numpy())
        final.append(fuse_views(result['score'],c).cpu().numpy())
    return np.stack(base),np.stack(final)


def initialize_environment():
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=False


def gpu_check():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if any(x.strip() and x.strip()!=str(os.getpid()) for x in p.stdout.splitlines()):raise RuntimeError('Another GPU process active')


def load_base(c):
    path=Path(c['base_checkpoint']);original=read(path.parent/'identity.json')
    base=torch.load(path,map_location='cpu',weights_only=False)
    completion=read(path.parent/'completion.json')
    if sha(path)!=completion['checkpoint_sha256'] or base['identity']!=original['digest']:
        raise ValueError('Base checkpoint identity mismatch')
    if base['choice'].get('normal_gate_weight',0):raise ValueError('Use the ungated original localizer')
    mismatches=[]
    for rel in ('localization_v3/data.py','localization_v3/model.py','models/paclip.py','config.py'):
        previous=original['source_sha256'].get(rel,original['source_sha256'].get(str(Path(rel))))
        if previous!=sha(ROOT/rel):mismatches.append(rel)
    proof=None
    if mismatches:
        if mismatches!=['localization_v3/data.py'] or not c.get('cache_compatibility_proof'):
            raise ValueError('Cached frozen extractor changed: '+','.join(mismatches))
        proof=read(c['cache_compatibility_proof'])
        previous=original['source_sha256'].get('localization_v3/data.py',original['source_sha256'].get(str(Path('localization_v3/data.py'))))
        if (proof.get('status')!='exactly_compatible' or proof.get('base_identity')!=original['digest'] or
            proof.get('old_data_source_sha256')!=previous or
            proof.get('current_data_source_sha256')!=sha(ROOT/'localization_v3/data.py') or
            not proof.get('all_sample_arrays_exact') or not proof.get('support_views_and_targets_exact')):
            raise ValueError('Invalid exact preprocessing compatibility proof')
    data=SliceSet(base['config']['manifest'],base['config']['support'])
    if data.identity!=original['data']:raise ValueError('Data identity mismatch')
    cache_path=path.parent/'support_validation_cache.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    if cache['identity']!=original['digest']:raise ValueError('Cache identity mismatch')
    if proof and (proof['data_identity_digest']!=digest(data.identity) or
                  proof['cache_sha256']!=sha(cache_path) or proof['samples_checked']!=len(data.rows)):
        raise ValueError('Compatibility proof does not bind this data/cache')
    allowed={r['file'] for r in data.splits['train']}
    if {m['sample'] for m in cache['metadata']}!=allowed:raise ValueError('Support cache mismatch')
    if any(i['row']['split']!='val' for i in cache['validation']):raise ValueError('Validation cache mismatch')
    if {i['row']['file'] for i in cache['validation']}!={r['file'] for r in data.splits['val']}:
        raise ValueError('Validation sample mismatch')
    c.update(crop_size=base['config']['crop_size'],alpha=base['choice']['alpha'],task=base['config'].get('task','brain_mri'),
             split_counts={s:len(rows) for s,rows in data.splits.items()},patient_disjoint_verified=data.patient_verified)
    return base,data,cache,cache_path


def fit(model,base,data,cache,c,out,identity,anchor):
    localizer_digest=state_digest(model.localizer.state_dict())
    optimizer=torch.optim.AdamW(model.foreground.parameters(),lr=c['lr'],weight_decay=1e-5)
    features=cache['train_features'].to(c['device'])
    targets={k:v.to(c['device']) for k,v in cache['targets'].items()}
    view_indices=collections.defaultdict(list)
    for i,m in enumerate(cache['metadata']):view_indices[m['sample']].append(i)
    normal=[r['file'] for r in data.splits['train'] if not r['label']]
    abnormal=[r['file'] for r in data.splits['train'] if r['label']]
    rng=np.random.default_rng(c['seed']);access=collections.Counter();history=[];best=None
    for step in range(1,c['steps']+1):
        model.train()
        samples=list(rng.choice(normal,c['batch_size']//2))+list(rng.choice(abnormal,c['batch_size']//2))
        indices=[int(rng.choice(view_indices[sample])) for sample in samples]
        idx=torch.as_tensor(indices,device=c['device'])
        logits=model.foreground(features[idx])
        result=foreground_loss(logits,{k:v[idx] for k,v in targets.items()},margin=c['margin'],
                               lesion_weight=c['lesion_weight'],hard_fraction=c['hard_fraction'])
        if not torch.isfinite(result['total']):raise FloatingPointError('Nonfinite foreground loss')
        optimizer.zero_grad(set_to_none=True);result['total'].backward()
        if any(p.grad is not None for p in model.localizer.parameters()):raise AssertionError('Localizer received gradients')
        torch.nn.utils.clip_grad_norm_(model.foreground.parameters(),5.);optimizer.step();access.update(samples)
        if step%c['validate_every']==0 or step==c['steps']:
            original,score=validation_maps(model,cache['validation'],c)
            choice,profiles=choose(score,cache['validation'],anchor,c)
            if state_digest(model.localizer.state_dict())!=localizer_digest:raise AssertionError('Localizer changed')
            item={'step':step,'loss':{k:float(v.detach()) for k,v in result.items()},'choice':choice}
            history.append(item)
            checkpoint={'format':'foreground_separated_v1','identity':identity,'head':base['head'],
                        'foreground':cpu_state(model.foreground),'step':step,'choice':choice,'config':c,
                        'localizer_state_sha256':localizer_digest,'trainable_localizer_parameters':0,
                        'trainable_encoder_parameters':0,'foreground_parameters':sum(p.numel() for p in model.foreground.parameters())}
            if best is None or tuple(choice['selection_key'])>tuple(best['choice']['selection_key']):
                best=checkpoint
                save_torch(out/'checkpoint_selected.pt',checkpoint)
                save_json(out/'selected_validation_profiles.json',profiles)
            save_torch(out/'checkpoint_last.pt',{**checkpoint,'optimizer':optimizer.state_dict(),
                       'rng':rng.bit_generator.state,'access':dict(access)})
            save_json(out/'training_history.json',history)
            save_json(out/'status.json',{'status':'training_foreground_only','step':step,'total':c['steps'],
                                        'eligible':choice['eligible']})
            print(f'step={step} eligible={choice["eligible"]} image_auc={choice["ranking"]["image_auroc"]:.4f} pixel_auc={choice["ranking"]["pixel_auroc"]:.4f} full_dice={choice["full"]["dice_micro"]:.4f} roi_dice={choice["roi"]["dice_micro"]:.4f}',flush=True)
    if set(access)!={r['file'] for r in data.splits['train']} or sum(access.values())!=c['steps']*c['batch_size']:
        raise AssertionError('Training support access mismatch')
    save_json(out/'training_access.json',dict(access))
    save_json(out/'frozen_localizer_audit.json',{'before':localizer_digest,
             'after':state_digest(model.localizer.state_dict()),'all_localizer_gradients_none':True,
             'trainable_localizer_parameters':0,'trainable_encoder_parameters':0})
    model.foreground.load_state_dict(best['foreground']);model.eval()
    return best


@torch.no_grad()
def evaluate(model,encoder,data,base,selected,c,out):
    names=('original','foreground');max_difference=0.
    baseline_threshold=base['choice']['metrics']['threshold'];threshold=selected['choice']['threshold']
    accumulators={name:{scope:OperatingAccumulator(t) for scope in ('full','roi')}
                  for name,t in [('original',baseline_threshold),('foreground',threshold)]}
    folder=out/'test_predictions';folder.mkdir()
    for number,row in enumerate(data.splits['test'],1):
        gray,_,gt,roi=data.read(row)
        empty=np.zeros_like(gray);all_pixels=np.ones_like(gray,dtype=bool)
        # Image views at inference are built WITHOUT passing test GT or ROI.
        images,_=make_views(gray,empty,empty,all_pixels,crop=c['crop_size'])
        features=extract_views(encoder,images,c['device']).to(c['device'])
        result=model(features)
        original=fuse_views(result['anomaly_probability'],c).cpu().numpy()
        final=fuse_views(result['score'],c).cpu().numpy()
        foreground=fuse_views(result['foreground_probability'],c).cpu().numpy()
        if not np.isfinite(final).all() or (final>original+2e-7).any():raise AssertionError('Invalid foreground-gated score')
        with np.load(Path(c['base_checkpoint']).parent/'test_predictions'/row['file']) as old:
            difference=float(np.max(np.abs(original-old['candidate_score'])))
            max_difference=max(max_difference,difference)
            if not np.allclose(original,old['candidate_score'],rtol=1e-5,atol=2e-5):raise AssertionError('Old score reproduction failed')
        np.savez_compressed(folder/row['file'],original_score=original,foreground_score=foreground,
                            score=final,prediction=final>=threshold,original_prediction=original>=baseline_threshold,gt=gt,roi=roi)
        for name,score in [('original',original),('foreground',final)]:
            accumulators[name]['full'].add(score,gt,all_pixels)
            accumulators[name]['roi'].add(score,gt,roi)
        if number%500==0:save_json(out/'status.json',{'status':'test_inference','done':number,'total':len(data.splits['test'])})
    metrics={};files=sorted(folder.glob('*.npz'))
    for name,key in [('original','original_score'),('foreground','score')]:
        save_json(out/'status.json',{'status':'exact_streaming_metrics','model':name,'test_count':len(files)})
        metrics[name]={'ranking':rank_npz(files,key),'full':accumulators[name]['full'].result(),
                       'roi':accumulators[name]['roi'].result()}
    save_json(out/'metrics.json',metrics)
    save_json(out/'prediction_audit.json',{'original_reproduction_max_abs_difference':max_difference,
              'test_count':len(data.splits['test']),'gt_and_roi_not_used_for_prediction':True,
              'hard_roi_mask_applied':False,'score_never_exceeds_original':True,
              'fusion':'multiply anomaly and learned foreground within each view, then original fixed view fusion'})
    make_gallery(data,out,metrics)
    return metrics


def make_gallery(data,out,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos=[];neg=[]
    for row in data.splits['test']:
        gray,_,gt,roi=data.read(row)
        (pos if gt.any() else neg).append((int(gt.sum()) if gt.any() else int(roi.sum()),row['file'],row))
    pos.sort(key=lambda a:a[:2]);neg.sort(key=lambda a:a[:2])
    chosen=[pos[int(q*(len(pos)-1))][2] for q in (.1,.5,.9)]+[neg[len(neg)//2][2]]
    for zoom,filename in [(False,'comparison.png'),(True,'localization_zoom.png')]:
        fig,axes=plt.subplots(4,4,figsize=(13,12),squeeze=False)
        for j,row in enumerate(chosen):
            gray,_,gt,roi=data.read(row)
            with np.load(out/'test_predictions'/row['file']) as d:
                for i in range(4):
                    ax=axes[i,j];ax.imshow(gray,cmap='gray',vmin=0,vmax=1);ax.axis('off')
                    if i in (1,3):
                        prefix='original_' if i==1 else ''
                        score=d[prefix+'score'];pred=d[prefix+'prediction']
                        ax.imshow(score,cmap='turbo',vmin=0,vmax=1,alpha=.35)
                        if pred.any() and not pred.all():ax.contour(pred,levels=[.5],colors=['red'],linewidths=.7)
                        den=int(pred.sum()+gt.sum());dice=2*int((pred&gt).sum())/den if den else 1.
                        title=('Frozen original' if i==1 else 'Learned foreground + original')
                        title+=f'\nFull Dice {dice:.3f}' if gt.any() else f'\nFull FP {100*pred.mean():.2f}%'
                    elif i==2:
                        ax.imshow(d['foreground_score'],cmap='viridis',vmin=0,vmax=1,alpha=.65)
                        title='Predicted foreground probability\nNo test ROI supplied'
                    else:title=row['file']+f'\nGT {int(gt.sum())} px'
                    if gt.any():ax.contour(gt,levels=[.5],colors=['lime'],linewidths=.7)
                    if zoom:
                        ys,xs=np.where(roi);ax.set_xlim(max(int(xs.min())-10,0),min(int(xs.max())+11,224))
                        ax.set_ylim(min(int(ys.max())+11,224),max(int(ys.min())-10,0))
                    ax.set_title(title,fontsize=8)
        suffix='; display crop only, predictions remain full maps' if zoom else '; full unmasked maps'
        fig.suptitle('Same held-out slices; green=GT, red=prediction'+suffix,fontsize=11)
        fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/filename,dpi=140);plt.close(fig)
    save_json(out/'display_samples.json',[r['file'] for r in chosen])


def report(c,out,selected,metrics):
    old=metrics['original'];new=metrics['foreground']
    prior_path=ROOT/'runs/liver_bmad_background_loss_k32_20261029/test_metrics.json'
    prior=read(prior_path)['background_0p05'] if c['task']=='liver_ct' and prior_path.exists() else None
    columns=[('旧密集头',old)]+([('上一轮共享头背景损失0.05（历史参照）',prior)] if prior else [])+[('冻结密集头＋独立前景头',new)]
    label={'brain_mri':'脑MRI','thymoma':'胸腺瘤CT','liver_ct':'Liver CT'}[c['task']]
    count=c['split_counts']
    lines=[f'# {label} K={count["train"]}：独立前景头试验','',
           f'同一{count["train"]}张支持图、{count["val"]}张验证图、{count["test"]}张测试图，seed=0。新增独立前景判别输出头，训练{c["steps"]}步；BiomedCLIP和原病灶定位头全部冻结，没有安装适配器。',
           '前景头仅使用冻结的第5/8/11层特征，约7.7万参数；不读取输入像素阈值、测试ROI或测试GT生成预测。训练时使用数据中既有的近似ROI作为弱前景标签，病灶GT覆盖ROI误差；2像素边界缓冲不作背景负例。',
           '损失分别归一前景/背景BCE，并增加病灶前景保留项。最终异常分数为每个视图的原病灶概率×预测前景概率，再按原固定权重融合；没有推理时的硬ROI置零。',
           '检查点与阈值仅由验证集选定，selection_frozen.json写入后才执行本轮测试。历史背景损失列仅作已有同测试集结果参照，训练策略和可训练参数不同，不是同预算消融。','',
           '| 指标 | '+' | '.join(name for name,m in columns)+' |','|---|'+'---:|'*len(columns)]
    for group,key,title in [('ranking','image_auroc','Image AUROC（全图最大值）'),('ranking','pixel_auroc','Pixel AUROC（全图）'),
                            ('ranking','pixel_ap','Pixel AP（全图）'),('full','dice_micro','Dice（全图）'),
                            ('full','small_lesion_recall','小病灶召回（全图预测）'),('full','normal_fp_area','正常全图误报面积')]:
        lines.append('| '+title+' | '+' | '.join(f'{m[group][key]*100:.4f}%' for name,m in columns)+' |')
    lines+=['','| ROI内定位保留检查 | '+' | '.join(name for name,m in columns)+' |','|---|'+'---:|'*len(columns)]
    for key,title in [('dice_micro','旧ROI约束Dice'),('small_lesion_recall','旧ROI约束小病灶召回'),('normal_fp_area','正常ROI误报面积')]:
        lines.append('| '+title+' | '+' | '.join(f'{m["roi"][key]*100:.4f}%' for name,m in columns)+' |')
    lines+=['','## 验证与冻结检查','',
            f'选中第{selected["step"]}步；验证保护通过：{selected["choice"]["eligible"]}；阈值：{selected["choice"]["threshold"]:.8g}。',
            '保护条件沿用上一轮：ROI Dice最多下降0.5个百分点、召回不下降、正常ROI误报最多增加0.1个百分点；全图正常误报≤1%，Image AUROC不下降、Pixel AUROC至少增加1个百分点。',
            '原病灶头参数更新数为0，编码器参数更新数为0；旧模型完整分数图与已保存预测逐像素核对。原训练入口和默认损失保持不变，新分支通过run_foreground.py显式运行。',
            '单图推理：python run_foreground.py --predict 图像.png --checkpoint 本目录/checkpoint_selected.pt --prediction-output 输出.npz；无需提供ROI掩码。','',
            '## 适用范围','',
            {'liver_ct':'BMAD输入本身已经按器官标注屏蔽肝外背景，本轮前景头学习的是这种预处理输入的有效组织；不能据此声称已能在原始完整腹部CT上自动分割肝脏。',
             'brain_mri':'沿用既有MRI K=5支持集及3715张测试划分。近似ROI来自图像强度前景；不是解剖学脑组织真值。该划分患者独立性未认证，病例编号前缀存在跨训练/测试候选池重合，不能作为患者独立泛化证明。',
             'thymoma':'沿用既有K=5胸腺瘤切片划分（原训练池1548张、验证220张、测试466张），梯度训练仅五张支持图；正常为病灶患者无病灶切片，没有NLST独立正常患者，不是早先585张测试协议。ROI是既有身体前景近似，不能当成胸腺或病灶器官真值。'}[c['task']],
            '原头冻结只保证其自身输出不变，最终相乘仍可能改变病灶排序与召回；必须看最终指标。单种子、少量源体积、已有测试集上的探索不等于论文复现或统计显著性。','',
            '## 同病例图','',
            '病例按GT面积10%、50%、90%位置与正常ROI面积中位数选取，不按预测成功与否挑图。第三行是模型预测的前景概率；绿色为病灶GT，红色为预测异常。','',
            '![完整分数图](comparison.png)','',
            '![仅显示时放大的定位图](localization_zoom.png)']
    (out/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@torch.no_grad()
def predict_image(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('format')!='foreground_separated_v1':raise ValueError('Not a foreground-head checkpoint')
    c=checkpoint['config'];gpu_check()
    encoder=FrozenEncoder().to(c['device']);model=ForegroundSeparatedModel(checkpoint['head']).to(c['device'])
    model.foreground.load_state_dict(checkpoint['foreground']);model.eval()
    with Image.open(args.predict) as im:
        gray=np.asarray(im.convert('L').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
    empty=np.zeros_like(gray);all_pixels=np.ones_like(gray,dtype=bool)
    images,_=make_views(gray,empty,empty,all_pixels,crop=c['crop_size'])
    result=model(extract_views(encoder,images,c['device']).to(c['device']))
    score=fuse_views(result['score'],c).cpu().numpy()
    foreground=fuse_views(result['foreground_probability'],c).cpu().numpy()
    dest=Path(args.prediction_output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,score=score,prediction=score>=checkpoint['choice']['threshold'],foreground_score=foreground)
    print(str(dest),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--predict')
    parser.add_argument('--checkpoint');parser.add_argument('--prediction-output',default='foreground_prediction.npz')
    args=parser.parse_args();initialize_environment()
    if args.predict:
        if not args.checkpoint:parser.error('--predict requires --checkpoint')
        return predict_image(args)
    if not args.config:parser.error('--config required for training')
    path=Path(args.config).resolve();c=read(path)
    for key in ('base_checkpoint','output','snapshot','cache_compatibility_proof'):
        if c.get(key):c[key]=str((path.parent/c[key]).resolve())
    c.setdefault('support_count',32)
    out=Path(c['output'])
    if out.exists():raise FileExistsError(out)
    if not (Path(c['snapshot'])/'code.zip').exists():raise FileNotFoundError('Required pre-edit snapshot')
    if c['steps']!=1200 or c['seed']!=0 or c['batch_size']!=8:raise ValueError('Use the registered pilot settings')
    gpu_check();started=time.time()
    base,data,cache,cache_path=load_base(c)
    if len(data.splits['train'])!=c['support_count']:raise ValueError('Configured support count mismatch')
    out.mkdir();save_json(out/'status.json',{'status':'preflight'})
    sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.relative_to(ROOT).parts}
    cache_hash=sha(cache_path);base_hash=sha(c['base_checkpoint'])
    proof_hash=sha(c['cache_compatibility_proof']) if c.get('cache_compatibility_proof') else None
    identity=digest({'config':c,'source':sources,'data':data.identity,'base_sha256':base_hash,'cache_sha256':cache_hash,
                     'compatibility_proof_sha256':proof_hash})
    save_json(out/'identity.json',{'digest':identity,'config':c,'source':sources,'data':data.identity,
                                  'base_sha256':base_hash,'cache_sha256':cache_hash,'compatibility_proof_sha256':proof_hash})
    save_json(out/'config.json',c)
    for rel in sources:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=ForegroundSeparatedModel(base['head']).to(c['device'])
    original,_=validation_maps(model,cache['validation'],c)
    items=cache['validation'];masks=[i['gt'] for i in items];rois=[i['roi'] for i in items]
    full=[np.ones_like(g,dtype=bool) for g in masks];threshold=base['choice']['metrics']['threshold']
    anchor={'threshold':threshold,'roi':operating_metrics(original,masks,rois,threshold),
            'full':operating_metrics(original,masks,full,threshold),'ranking':standard_ranking_metrics(original,masks,rois)}
    save_json(out/'original_validation.json',anchor)
    selected=fit(model,base,data,cache,c,out,identity,anchor)
    frozen={'identity':identity,'checkpoint_sha256':sha(out/'checkpoint_selected.pt'),
            'step':selected['step'],'choice':selected['choice'],'selection_source':'validation_only'}
    save_json(out/'selection_frozen.json',frozen)
    del cache;torch.cuda.empty_cache()
    save_json(out/'status.json',{'status':'final_test','selected_step':selected['step']})
    encoder=FrozenEncoder().to(c['device']);encoder.eval()
    if any(p.requires_grad for p in encoder.parameters()):raise AssertionError('Encoder not frozen')
    metrics=evaluate(model,encoder,data,base,selected,c,out)
    report(c,out,selected,metrics)
    save_json(out/'completion.json',{'status':'complete','identity':identity,'steps':c['steps'],
              'selected_step':selected['step'],'test_count':len(data.splits['test']),
              'checkpoint_sha256':sha(out/'checkpoint_selected.pt'),'metrics_sha256':sha(out/'metrics.json'),
              'selection_sha256':sha(out/'selection_frozen.json'),'elapsed_seconds':time.time()-started,
              'validation_guard_passed':selected['choice']['eligible']})
    save_json(out/'status.json',{'status':'complete'})
    print(json.dumps({'status':'complete','metrics':metrics},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
