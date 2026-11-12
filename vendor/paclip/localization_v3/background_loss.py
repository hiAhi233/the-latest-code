"""Optional region-balanced background supervision; legacy loss stays intact."""
import torch
import torch.nn.functional as F
from .loss import recall_loss


def balanced_background_loss(logits,target,*,weight=0.,samples=512,margin=2,recall_weight=.5):
    """GT overrides approximate ROI; exclude a boundary buffer from negatives.

    Half uniform and half hard negatives, normalized per view, then per batch.
    Weight zero is exactly the old objective, including its gradients.
    ROI is used only in training; inference scores are never masked here.
    """
    if weight<0 or samples<1 or margin<0:raise ValueError('Invalid background loss settings')
    result=recall_loss(logits,target,recall_weight=recall_weight)
    if weight==0:return result
    z=logits.float()
    occupied=target['roi'].to(z.device).bool() | (target['hard'].to(z.device)>0) | (target['soft'].to(z.device)>0)
    if margin:
        occupied=F.max_pool2d(occupied[:,None].float(),2*margin+1,stride=1,padding=margin)[:,0].bool()
    terms=[]
    for i in range(len(z)):
        values=F.softplus(z[i][~occupied[i]])
        if not values.numel():continue
        k=min(samples,values.numel())
        random=values[torch.randperm(values.numel(),device=z.device)[:k]]
        terms.append(.5*random.mean()+.5*values.topk(k).values.mean())
    background=torch.stack(terms).mean() if terms else z.sum()*0
    return {**result,'total':result['total']+weight*background,'background':background}
