"""Matched warm-start ablation, frozen features, validation-only selection."""
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
from .background_loss import balanced_background_loss
from .data import SliceSet,sha,digest,make_views,extract_views
from .model import RecallDenseHead,FrozenEncoder,combine_views,view_boxes
from .metrics import operating_metrics,standard_ranking_metrics,threshold_candidates
from .experiment import ROOT,save_json,save_torch,cpu_state,predict_cached


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def maps_for(head,items,c):
    a,b=predict_cached(head,items,c['device'],c['crop_size'])
    return (1-c['alpha'])*a+c['alpha']*b


def choose(scores,items,anchor,c):
    if any(i['row']['split']!='val' for i in items):raise ValueError('Validation only')
    masks=[i['gt'] for i in items];rois=[i['roi'] for i in items]
    full=[np.ones_like(g,dtype=bool) for g in masks]
    ranking=standard_ranking_metrics(scores,masks,rois)
    thresholds=set(threshold_candidates(scores,masks,rois)+threshold_candidates(scores,masks,full))
    thresholds.add(anchor['threshold'])
    best=None;profiles=[]
    for threshold in sorted(thresholds):
        roi=operating_metrics(scores,masks,rois,threshold)
        grid=operating_metrics(scores,masks,full,threshold)
        d=max(0,anchor['roi']['dice_micro']-roi['dice_micro'])
        r=max(0,anchor['roi']['small_lesion_recall']-roi['small_lesion_recall'])
        f=max(0,roi['normal_fp_area']-anchor['roi']['normal_fp_area'])
        local=d<=c['dice_tolerance']+1e-12 and r<=c['recall_tolerance']+1e-12 and f<=c['normal_fp_tolerance']+1e-12
        background=grid['normal_fp_area']<=c['max_full_normal_fp']+1e-12
        auc=(ranking['image_auroc']+1e-12>=anchor['ranking']['image_auroc'] and
             ranking['pixel_auroc']>=anchor['ranking']['pixel_auroc']+c['min_pixel_auc_gain'])
        eligible=local and background and auc
        quality=.5*(ranking['pixel_auroc']+ranking['image_auroc'])+.1*grid['dice_micro']
        quality-=2*d+2*r+5*f+max(0,grid['normal_fp_area']-c['max_full_normal_fp'])
        key=(int(eligible),quality,grid['dice_micro'],roi['dice_micro'])
        row={'threshold':float(threshold),'full':grid,'roi':roi,'ranking':ranking,
             'localization_guard':local,'background_guard':background,'auc_guard':auc,
             'eligible':eligible,'selection_key':key}
        profiles.append(row)
        if best is None or key>best['selection_key']:best=row
    return best,profiles


def train_arm(weight,cache,base,anchor,c,out,identity):
    name='old_loss_control' if weight==0 else 'background_'+str(weight).replace('.','p')
    folder=out/name;folder.mkdir()
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    head=RecallDenseHead().to(c['device']);head.load_state_dict(base['head'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['lr'],weight_decay=1e-5)
    features=cache['train_features'].to(c['device'])
    targets={k:v.to(c['device']) for k,v in cache['targets'].items()}
    positive=[i for i,m in enumerate(cache['metadata']) if m['positive']]
    negative=[i for i,m in enumerate(cache['metadata']) if not m['positive']]
    rng=np.random.default_rng(c['seed']);access=collections.Counter();history=[];best=None
    for step in range(1,c['steps']+1):
        head.train();idx=np.r_[rng.choice(positive,4),rng.choice(negative,4)]
        ix=torch.as_tensor(idx,device=c['device'])
        result=balanced_background_loss(head(features[ix]),{k:v[ix] for k,v in targets.items()},
              weight=weight,samples=c['samples'],margin=c['margin'],recall_weight=c['recall_weight'])
        if not torch.isfinite(result['total']):raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);result['total'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        for i in idx:access[cache['metadata'][int(i)]['sample']]+=1
        if step%c['validate_every']==0 or step==c['steps']:
            scores=maps_for(head,cache['validation'],c)
            choice,profiles=choose(scores,cache['validation'],anchor,c)
            history.append({'step':step,'loss':{k:float(v.detach()) for k,v in result.items()},'choice':choice})
            if best is None or tuple(choice['selection_key'])>tuple(best['choice']['selection_key']):
                best={'format':'background_loss_ablation_v1','identity':identity,'head':cpu_state(head),
                      'step':step,'base_step':base['step'],'weight':weight,'choice':choice,
                      'config':c,'trainable_encoder_parameters':0}
                save_torch(folder/'checkpoint_selected.pt',best)
                save_json(folder/'selected_validation_profiles.json',profiles)
            save_json(folder/'training_history.json',history)
            save_json(out/'status.json',{'status':'training','arm':name,'step':step,'total_steps_per_arm':c['steps']})
            print(f'{name} step={step} eligible={choice["eligible"]} image_auc={choice["ranking"]["image_auroc"]:.4f} pixel_auc={choice["ranking"]["pixel_auroc"]:.4f} roi_dice={choice["roi"]["dice_micro"]:.4f}',flush=True)
    assert set(access)=={m['sample'] for m in cache['metadata']} and sum(access.values())==c['steps']*8
    save_json(folder/'training_access.json',dict(access))
    save_json(folder/'training_completion.json',{'steps':c['steps'],'selected_step':best['step'],'weight':weight,
                                               'checkpoint_sha256':sha(folder/'checkpoint_selected.pt')})
    best['name']=name
    del head,optimizer,features,targets;torch.cuda.empty_cache()
    return best


@torch.no_grad()
def test_all(base,selected,data,c,out):
    encoder=FrozenEncoder().to(c['device']);encoder.eval()
    if any(p.requires_grad for p in encoder.parameters()):raise AssertionError('Encoder trainable')
    states=[{'name':'original_k32','head':base['head'],'choice':{'threshold':base['choice']['metrics']['threshold']}}]+selected
    heads={s['name']:RecallDenseHead().to(c['device']) for s in states}
    for state in states:heads[state['name']].load_state_dict(state['head']);heads[state['name']].eval()
    scores={s['name']:[] for s in states};masks=[];rois=[];rows=[];max_difference=0.
    for row in data.splits['test']:
        gray,soft,gt,roi=data.read(row)
        images,_=make_views(gray,soft,gt,roi,crop=c['crop_size'])
        features=extract_views(encoder,images,c['device']).to(c['device'])
        masks.append(gt);rois.append(roi);rows.append(row)
        for state in states:
            name=state['name'];p=heads[name](features).sigmoid()
            a,b=combine_views(p,view_boxes(crop=c['crop_size']))
            score=((1-c['alpha'])*a+c['alpha']*b).cpu().numpy()
            scores[name].append(score)
            folder=out/name/'test_predictions';folder.mkdir(parents=True,exist_ok=True)
            threshold=state['choice']['threshold']
            np.savez_compressed(folder/row['file'],score=score,prediction=score>=threshold,gt=gt,roi=roi)
            if name=='original_k32':
                with np.load(Path(c['base_checkpoint']).parent/'test_predictions'/row['file']) as old:
                    difference=float(np.max(np.abs(score-old['candidate_score'])))
                    max_difference=max(max_difference,difference)
                    if not np.allclose(score,old['candidate_score'],rtol=1e-5,atol=2e-5):
                        raise AssertionError('Original model prediction changed')
    results={};full=[np.ones_like(g,dtype=bool) for g in masks]
    for state in states:
        name=state['name'];threshold=state['choice']['threshold']
        results[name]={'ranking':standard_ranking_metrics(scores[name],masks,rois),
                      'full':operating_metrics(scores[name],masks,full,threshold),
                      'roi':operating_metrics(scores[name],masks,rois,threshold)}
        save_json(out/name/'metrics.json',results[name])
    save_json(out/'test_metrics.json',results)
    save_json(out/'prediction_audit.json',{'original_reproduction_max_abs_difference':max_difference,
             'test_count':len(rows),'full_maps_not_roi_masked':True,'common_encoder':True})
    make_gallery(states,rows,data,c,out,results)
    return results


def make_gallery(states,rows,data,c,out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos=[];neg=[]
    for row in rows:
        gray,_,gt,roi=data.read(row)
        (pos if gt.any() else neg).append((int(gt.sum()) if gt.any() else int(roi.sum()),row['file'],row))
    pos.sort(key=lambda a:a[:2]);neg.sort(key=lambda a:a[:2])
    chosen=[pos[int(q*(len(pos)-1))][2] for q in (.1,.5,.9)]+[neg[len(neg)//2][2]]
    fig,axes=plt.subplots(len(states)+1,4,figsize=(13,3*(len(states)+1)),squeeze=False)
    for j,row in enumerate(chosen):
        gray,_,gt,roi=data.read(row)
        for i in range(len(states)+1):
            ax=axes[i,j];ax.imshow(gray,cmap='gray',vmin=0,vmax=1);ax.axis('off')
            if gt.any():ax.contour(gt,levels=[.5],colors=['lime'],linewidths=.7)
        axes[0,j].set_title(row['file']+f'\nGT {int(gt.sum())} px',fontsize=8)
        for i,state in enumerate(states,1):
            name=state['name']
            with np.load(out/name/'test_predictions'/row['file']) as p:
                score=p['score'];pred=p['prediction'];den=int(pred.sum()+gt.sum())
                dice=2*int((pred&gt).sum())/den if den else 1.
                axes[i,j].imshow(score,cmap='turbo',vmin=0,vmax=1,alpha=.35)
                if pred.any() and not pred.all():axes[i,j].contour(pred,levels=[.5],colors=['red'],linewidths=.7)
                title=f'{name}\nFull Dice {dice:.3f}' if gt.any() else f'{name}\nFull FP {100*pred.mean():.2f}%'
                axes[i,j].set_title(title,fontsize=8)
    fig.suptitle('Same held-out slices; full unmasked scores; green=GT, red=prediction',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    save_json(out/'display_samples.json',[r['file'] for r in chosen])


def write_report(c,out,selected,validation_decision,results):
    labels={'original_k32':'原K=32权重','old_loss_control':'继续旧损失（同预算）',
            'background_0p02':'背景权重0.02','background_0p05':'背景权重0.05'}
    names=list(results)
    lines=['# Liver CT K=32：新增背景损失对照','',
           '原始代码快照：'+c['snapshot'],
           '相同32张支持图、66张验证图、189张测试图；从同一第600步旧权重出发，每组继续600步、lr=1e-5，seed=0。',
           '旧损失继续训练是同预算对照；新增背景损失按每个视图单独取平均，512个随机负像素＋512个高分负像素，各占一半；背景权重分别0.02和0.05，ROI/GT外留2像素缓冲。',
           '训练、数据预处理、主干和推理结构一致；仅训练新增损失时使用区域分组，最终分数图不做ROI置零。正常参照模块未加入本轮。',
           '每组训练检查点与阈值由验证集选择；所有组完成选模并写入selection_frozen.json后，才执行本轮测试。','',
           '| 指标 | '+' | '.join(labels[n] for n in names)+' |',
           '|---|'+'---:|'*len(names)]
    for group,key,title in [('ranking','image_auroc','Image AUROC（全图最大值）'),('ranking','pixel_auroc','Pixel AUROC（全图）'),
                            ('ranking','pixel_ap','Pixel AP（全图）'),('full','dice_micro','Dice（全图预测）'),
                            ('full','small_lesion_recall','小病灶召回（全图预测）'),('full','normal_fp_area','正常误报面积（全图）')]:
        lines.append('| '+title+' | '+' | '.join(f'{100*results[n][group][key]:.4f}%' for n in names)+' |')
    lines+=['','原报告约26.86%的Dice使用了ROI限制预测。上表改为同一张完整连续分数图按验证阈值二值化后的全图Dice；旧权重也按相同规则重算，不能把这两种Dice混用。','',
            '| 附加定位保护指标 | '+' | '.join(labels[n] for n in names)+' |','|---|'+'---:|'*len(names)]
    for key,title in [('dice_micro','ROI约束Dice（旧口径）'),('small_lesion_recall','ROI约束小病灶召回'),('normal_fp_area','正常ROI误报面积')]:
        lines.append('| '+title+' | '+' | '.join(f'{100*results[n]["roi"][key]:.4f}%' for n in names)+' |')
    lines+=['','## 验证选模与保留策略','',
            '试验前固定保护条件：相对旧起点，ROI Dice最多下降0.5个百分点，小病灶召回不下降，正常ROI误报最多增加0.1个百分点；全图正常误报≤1%，Image AUROC不下降，Pixel AUROC至少增加1个百分点。',
            '通过保护的候选优先，再按验证AUROC综合分选模；若无候选通过，保留得分最高的试验检查点用于如实报告，但不切换默认损失。',
            '验证阶段决定：'+json.dumps(validation_decision,ensure_ascii=False),
            '原loss.py和run_localization.py维持旧默认；新损失仅通过run_background_loss.py显式启用，代码、权重和结果全部保留。','',
            '## 同病例图','',
            '阳性按GT面积10%、50%、90%分层选择；阴性取ROI面积中位切片，不按预测成败挑图。显示完整分数图，未隐藏肝外背景。','',
            '![定位与背景对照](comparison.png)','',
            '这是同一小批数据上的单种子探索；数据划分与论文不同，不作论文复现或统计显著性结论。']
    (out/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    path=Path(args.config).resolve();c=read(path)
    for key in ('base_checkpoint','output','snapshot'):c[key]=str((path.parent/c[key]).resolve())
    if c['weights']!=[0.,.02,.05] or c['steps']!=600:raise ValueError('Use preregistered ablation config')
    out=Path(c['output'])
    if out.exists():raise FileExistsError(out)
    if not (Path(c['snapshot'])/'code.zip').is_file():raise FileNotFoundError('Required code snapshot')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if any(x.strip() and x.strip()!=str(os.getpid()) for x in p.stdout.splitlines()):raise RuntimeError('Another GPU process active')
    started=time.time();base_path=Path(c['base_checkpoint']);original=read(base_path.parent/'identity.json')
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    done=read(base_path.parent/'completion.json')
    if sha(base_path)!=done['checkpoint_sha256'] or base['identity']!=original['digest']:raise ValueError('Base checkpoint identity mismatch')
    if base['choice'].get('normal_gate_weight',0):raise ValueError('Use the ungated base head')
    for rel in ('localization_v3/data.py','localization_v3/model.py','models/paclip.py','config.py'):
        old=original['source_sha256'].get(rel,original['source_sha256'].get(str(Path(rel))))
        if old!=sha(ROOT/rel):raise ValueError('Frozen features or target generation changed: '+rel)
    data=SliceSet(base['config']['manifest'],base['config']['support'])
    if data.identity!=original['data']:raise ValueError('Base data changed')
    c.update(crop_size=base['config']['crop_size'],alpha=base['choice']['alpha'],recall_weight=base['config']['recall_weight'])
    if len(data.splits['train'])!=32:raise ValueError('Expected K=32')
    cache_path=base_path.parent/'support_validation_cache.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    if cache['identity']!=original['digest']:raise ValueError('Cache identity mismatch')
    allowed={r['file'] for r in data.splits['train']}
    if {m['sample'] for m in cache['metadata']}!=allowed:raise ValueError('Cache support mismatch')
    if any(i['row']['split']!='val' for i in cache['validation']):raise ValueError('Held-out cache misuse')
    out.mkdir();save_json(out/'status.json',{'status':'preflight'})
    sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.relative_to(ROOT).parts}
    identity=digest({'config':c,'sources':sources,'data':data.identity,'base_sha256':sha(base_path),'cache_sha256':sha(cache_path)})
    save_json(out/'identity.json',{'digest':identity,'config':c,'sources':sources,'data':data.identity,
                                  'base_sha256':sha(base_path),'cache_sha256':sha(cache_path),'seed':c['seed']})
    save_json(out/'config.json',c)
    for rel in sources:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    head=RecallDenseHead().to(c['device']);head.load_state_dict(base['head']);head.eval()
    scores=maps_for(head,cache['validation'],c);items=cache['validation']
    masks=[i['gt'] for i in items];rois=[i['roi'] for i in items];full=[np.ones_like(g,dtype=bool) for g in masks]
    threshold=base['choice']['metrics']['threshold']
    anchor={'threshold':threshold,'roi':operating_metrics(scores,masks,rois,threshold),
            'full':operating_metrics(scores,masks,full,threshold),'ranking':standard_ranking_metrics(scores,masks,rois)}
    save_json(out/'original_validation.json',anchor)
    del head;torch.cuda.empty_cache()
    selected=[train_arm(w,cache,base,anchor,c,out,identity) for w in c['weights']]
    eligible=[s for s in selected if s['weight']>0 and s['choice']['eligible']]
    winner=max(eligible,key=lambda s:tuple(s['choice']['selection_key'])) if eligible else None
    control=selected[0]
    if winner is not None and tuple(winner['choice']['selection_key'])<=tuple(control['choice']['selection_key']):winner=None
    decision={'recommend_new_loss_on_validation':winner is not None,'candidate':winner['name'] if winner else 'keep_old_loss',
              'default_main_entrypoint':'old_loss','selection_source':'validation_only'}
    frozen={'decision':decision,'models':{s['name']:{'checkpoint_sha256':sha(out/s['name']/'checkpoint_selected.pt'),
             'step':s['step'],'choice':s['choice']} for s in selected}}
    save_json(out/'selection_frozen.json',frozen)
    save_json(out/'status.json',{'status':'final_test','decision':decision})
    del cache;torch.cuda.empty_cache()
    results=test_all(base,selected,data,c,out)
    write_report(c,out,selected,decision,results)
    save_json(out/'completion.json',{'status':'complete','elapsed_seconds':time.time()-started,
              'test_count':len(data.splits['test']),'identity':identity,'decision':decision,
              'test_metrics_sha256':sha(out/'test_metrics.json'),'selection_sha256':sha(out/'selection_frozen.json')})
    save_json(out/'status.json',{'status':'complete'})
    print(json.dumps({'status':'complete','decision':decision,'results':results},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
