import math
import numpy as np
from scipy.ndimage import label
from scipy.stats import rankdata


def image_pool_scores(scores, rois):
    """Fixed pooling rules: top-200 matches the existing SR2 reference code.

    Max and top-5% are supplementary; test labels never select a pooling rule.
    """
    pooled={'top200':[], 'max':[], 'top5pct':[]}
    if len(scores)!=len(rois):raise ValueError('Image/ROI count mismatch')
    for score,roi in zip(scores,rois):
        score=np.asarray(score);roi=np.asarray(roi,bool)
        if score.shape!=roi.shape or not roi.any():raise ValueError('Invalid image ROI')
        values=score[roi]
        if not np.isfinite(values).all():raise ValueError('Nonfinite image scores')
        pooled['max'].append(float(values.max()))
        for name,k in [('top200',min(200,values.size)),('top5pct',max(1,math.ceil(values.size*.05)))]:
            pooled[name].append(float(np.partition(values,values.size-k)[-k:].mean()))
    return pooled


def standard_ranking_metrics(scores, masks, rois):
    """Medical AA-CLIP/MadCLIP scoring: full-map max and all-grid pixel AUC.

    The existing ROI-masked and ROI-only pixel metrics are separate outputs.
    No retraining, operating threshold or test-based parameter choice is used.
    """
    labels=np.asarray([bool(np.asarray(g).any()) for g in masks])
    if len(scores)!=len(labels):raise ValueError('Image/label count mismatch')
    positive=int(labels.sum());negative=len(labels)-positive
    def auc(values):
        if not positive or not negative:return None
        ranks=rankdata(values,method='average')
        return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))
    pooled=image_pool_scores(scores,rois)
    full=ranking_metrics(scores,masks,[np.ones_like(r,dtype=bool) for r in rois])
    return {'image_auroc':auc([float(np.asarray(s).max()) for s in scores]),
            'image_auroc_roi_top200':auc(pooled['top200']),
            'image_auroc_max':auc(pooled['max']),
            'image_auroc_top5pct':auc(pooled['top5pct']),
            'pixel_auroc':full['pixel_auroc_roi_masked'],
            'pixel_ap':full['pixel_ap_roi_masked']}


def lesion_counts(pred, gt):
    """8-connected GT; one prediction component must cover >=10% of a lesion."""
    gt = np.asarray(gt, bool)
    pred = np.asarray(pred, bool)
    cc, n = label(gt, np.ones((3,3)))
    pc, _ = label(pred, np.ones((3,3)))
    detected = small = small_detected = 0
    for k in range(1, n+1):
        inside = cc == k
        area = int(inside.sum())
        overlap = np.bincount(pc[inside])[1:]
        hit = bool(len(overlap) and overlap.max() >= max(1, math.ceil(.1*area)))
        is_small = area <= gt.size*.01
        detected += hit
        small += is_small
        small_detected += hit and is_small
    return dict(lesions=n, detected=detected, small_lesions=small, small_detected=small_detected)


def operating_metrics(scores, masks, rois, threshold):
    tp = fp = fn = 0
    totals = dict(lesions=0, detected=0, small_lesions=0, small_detected=0)
    normal_fp, positive_dice = [], []
    for s,g,r in zip(scores,masks,rois):
        g,r = np.asarray(g,bool),np.asarray(r,bool)
        pred = (np.asarray(s) >= threshold) & r
        inter = int((pred & g).sum())
        tp += inter; fp += int((pred & ~g).sum()); fn += int((~pred & g).sum())
        if g.any():
            c = lesion_counts(pred,g)
            for k,v in c.items(): totals[k] += v
            positive_dice.append(2*inter/(int(pred.sum())+int(g.sum())))
        elif r.any():
            normal_fp.append(float(pred.sum()/r.sum()))
    return {'dice_micro':2*tp/max(2*tp+fp+fn,1),
            'positive_macro_dice':float(np.mean(positive_dice)) if positive_dice else None,
            'lesion_recall':totals['detected']/totals['lesions'] if totals['lesions'] else None,
            'small_lesion_recall':totals['small_detected']/totals['small_lesions'] if totals['small_lesions'] else None,
            'normal_fp_area':float(np.mean(normal_fp)) if normal_fp else None,
            'counts':totals,'pixels':{'tp':tp,'fp':fp,'fn':fn},'threshold':float(threshold)}


def ranking_metrics(scores, masks, rois):
    """Exact AP/AUC; ROI-excluded pixels form the lowest-score tie group.

    No score binning. Also return true ROI AP separately, keeping GT outside ROI
    in the full-image denominator.
    """
    s = np.concatenate([np.asarray(a)[r] for a,r in zip(scores,rois)])
    y = np.concatenate([g[r] for g,r in zip(masks,rois)]).astype(bool)
    if not np.isfinite(s).all(): raise ValueError('Nonfinite predictions')
    total_p = sum(int(g.sum()) for g in masks)
    total_n = sum(g.size for g in masks)-total_p
    inside_p = int(y.sum()); outside_p = total_p-inside_p
    outside_n = sum(int((~r).sum()) for r in rois)-outside_p
    order = np.argsort(-s, kind='mergesort')
    s,y = s[order],y[order]
    del order
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    del s
    cumulative = np.cumsum(y,dtype=np.int64)
    tp = cumulative[ends]; selected = ends+1
    del cumulative,y,ends
    dtp = np.diff(np.r_[0,tp]); dfp = np.diff(np.r_[0,selected-tp])
    ap_sum = float(np.sum(dtp.astype(np.float64)*tp/selected))
    concordant = float(np.sum(dfp.astype(np.float64)*(tp-.5*dtp)))
    inside_n = int(selected[-1])-inside_p
    roi_auc = concordant/(inside_p*inside_n) if inside_p and inside_n else None
    ap = ap_sum/total_p if total_p else 0.
    if outside_p: ap += outside_p/(total_p+total_n)
    concordant += outside_n*(inside_p+.5*outside_p)
    return {'pixel_ap_roi_masked':ap,
            'pixel_ap_true_roi':ap_sum/inside_p if inside_p else 0.,
            'pixel_auroc_true_roi':roi_auc,
            'pixel_auroc_roi_masked':concordant/(total_p*total_n) if total_p and total_n else None}


def threshold_candidates(scores, masks, rois):
    values = np.concatenate([s[r] for s,r in zip(scores,rois)])
    truth = np.concatenate([g[r] for g,r in zip(masks,rois)])
    order = np.argsort(-values,kind='mergesort')
    v = values[order]; y = truth[order]
    ends = np.flatnonzero(np.r_[v[1:] != v[:-1],True])
    tp = np.cumsum(y,dtype=np.int64)[ends]
    dice = 2*tp/(ends+1+sum(int(g.sum()) for g in masks))
    best = float(v[ends[int(dice.argmax())]])
    normals = [s[r] for s,g,r in zip(scores,masks,rois) if not g.any()]
    candidates = list(np.linspace(.025,.975,20)) + [best]
    if normals:
        n = np.concatenate(normals)
        candidates += list(np.quantile(n,[.99,.995,.999,1]))
        candidates += [float(np.nextafter(np.float32(n.max()),np.float32(np.inf)))]
    candidates += list(np.quantile(values,[.5,.9,.95,.98,.99,.995]))
    return sorted(set(float(x) for x in candidates))


def selection_key(metrics, reference, max_normal_fp=1.):
    """All three requested objectives must be non-inferior on validation.

    If none qualifies, rank provisional candidates but report guard failure.
    The test split never participates in this decision.
    """
    d = metrics['dice_micro']; r = metrics['small_lesion_recall']; f = metrics['normal_fp_area']
    if r is None or f is None: raise ValueError('Validation needs small lesions and normal images')
    safe = f <= max_normal_fp+1e-12
    passed = safe and d+1e-12 >= reference['dice_micro'] and r+1e-12 >= reference['small_lesion_recall'] and f <= reference['normal_fp_area']+1e-12
    deficit = max(0,reference['dice_micro']-d)+max(0,reference['small_lesion_recall']-r)+10*max(0,f-reference['normal_fp_area'])
    quality = .5*d+.3*r+.2*metrics['lesion_recall']-2*f
    # A degenerate reference must never make an all-ROI prediction preferable
    # to an operating point satisfying the preregistered absolute FP budget.
    if not safe: return (0, -1e6-f)
    return (int(passed), quality if passed else quality-deficit)


def select_validation(global_maps, zoom_maps, masks, rois, reference, rows, alphas=(0.,.25,.5), max_normal_fp=1.):
    if any(r['split'] != 'val' for r in rows):
        raise ValueError('Only validation data may select weights or thresholds')
    best = None; records = []
    for alpha in alphas:
        scores = (1-alpha)*global_maps + alpha*zoom_maps
        for threshold in threshold_candidates(scores,masks,rois):
            m = operating_metrics(scores,masks,rois,threshold)
            key = selection_key(m,reference,max_normal_fp)
            row = {'alpha':alpha,'metrics':m,'guard_passed':bool(key[0]),'selection_score':key[1]}
            records.append(row)
            if best is None or key > best[0]: best = (key,row)
    return best[1],records
