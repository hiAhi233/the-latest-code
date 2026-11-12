import torch
import torch.nn.functional as F


def recall_loss(logits, target, normal_weight=.25, recall_weight=.5):
    """Per-image area supervision and equal-connected-lesion coverage pressure.

    Small components receive equal total recall weight rather than losing to a
    large lesion in the same image. This term uses hard-preserved GT; the pixel
    and Dice terms use soft area targets. Clean normal images remain supervised.
    """
    z = logits.float()
    p = z.sigmoid()
    soft, hard, roi, lw = [target[k].to(z.device) for k in ('soft','hard','roi','lesion_weights')]
    roi = roi.bool()
    if not roi.flatten(1).any(1).all():
        raise ValueError('Empty training ROI')
    positive = hard.flatten(1).sum(1) > 0
    focal = -.75*soft*(1-p).square()*F.logsigmoid(z) - .25*(1-soft)*p.square()*F.logsigmoid(-z)
    per_pixel = (focal*roi).flatten(1).sum(1)/roi.flatten(1).sum(1)
    truth, prob = soft*roi, p*roi
    dice = 1-(2*(truth*prob).flatten(1).sum(1)+1e-6)/(truth.flatten(1).sum(1)+prob.flatten(1).sum(1)+1e-6)
    coverage = ((1-p)*lw*roi).flatten(1).sum(1)/(lw*roi).flatten(1).sum(1).clamp_min(1e-6)
    boundary = torch.zeros_like(per_pixel)
    for axis in (-1,-2):
        valid = roi.narrow(axis,1,roi.size(axis)-1) & roi.narrow(axis,0,roi.size(axis)-1)
        err = (torch.diff(p,dim=axis)-torch.diff(soft,dim=axis)).abs()
        boundary += (err*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
    zero = z.sum()*0
    real = (per_pixel+dice+.05*boundary+recall_weight*coverage)[positive].mean() if positive.any() else zero
    # Include full-ROI normal supervision; hard negatives are not pseudo lesions.
    normal_terms = []
    for i in torch.where(~positive)[0]:
        v = focal[i][roi[i]]
        normal_terms.append(.5*v.mean()+.5*v.topk(max(1,int(v.numel()*.05))).values.mean())
    normal = torch.stack(normal_terms).mean() if normal_terms else zero
    return {'total': real+normal_weight*normal, 'real':real, 'normal':normal,
            'lesion_coverage':coverage[positive].mean() if positive.any() else zero}
