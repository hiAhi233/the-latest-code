"""Optional normal-support reference, fitted without real abnormal/test features."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def build_memory(cache, normal_ids, centers=32):
    features=[]
    for i,meta in enumerate(cache['metadata']):
        if meta['sample'] not in normal_ids: continue
        f=cache['train_features'][i].float()
        roi=F.adaptive_max_pool2d(cache['targets']['roi'][i][None,None].float(),f.shape[-2:])[0,0].bool()
        features.append(f[:,:,roi].permute(0,2,1))
    if not features: raise ValueError('Normal support features required')
    x=F.normalize(torch.cat(features,dim=1),dim=-1)
    k=min(centers,x.shape[1]);result=[]
    for level in x:
        c=level[torch.linspace(0,len(level)-1,k).long()].clone()
        for _ in range(12):
            assignment=(level@c.T).argmax(1)
            updated=c.clone()
            for j in range(k):
                if (assignment==j).any(): updated[j]=F.normalize(level[assignment==j].mean(0),dim=0)
            c=updated
        result.append(c)
    return torch.stack(result)


@torch.no_grad()
def distance_maps(features,memory):
    f=F.normalize(features.float(),dim=2)
    similarity=torch.einsum('blchw,lkc->blkhw',f,memory.to(f.device))
    distance=(1-similarity.max(2).values).mean(1).clamp(0,2)
    return F.interpolate(distance[:,None],(224,224),mode='bilinear',align_corners=False)[:,0]


def fit_normal_cdf(distance,masks,rois,rows):
    if any(r['split']!='val' for r in rows): raise ValueError('Normal calibration requires validation')
    values=[d[r] for d,g,r in zip(distance,masks,rois) if not g.any()]
    if not values: raise ValueError('No normal validation images')
    quantiles=np.linspace(0,1,4097)
    knots=np.quantile(np.concatenate(values),quantiles)
    # Keep the right endpoint of repeated quantiles, preserving monotonicity.
    unique,indices=np.unique(knots[::-1],return_index=True)
    ranks=quantiles[len(knots)-1-indices]
    return {'knots':unique.tolist(),'ranks':ranks.tolist(),'fit_source':'normal_validation_only'}


def gate_scores(probability,distance,cdf,weight):
    if not 0<=weight<=1: raise ValueError('Invalid normal gate weight')
    percentile=np.interp(distance,cdf['knots'],cdf['ranks'],left=0.,right=1.)
    return (np.asarray(probability)*(1-weight+weight*percentile)).astype(np.float32)
