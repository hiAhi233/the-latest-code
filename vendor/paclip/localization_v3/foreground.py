"""Learned foreground output head; no adapter or inference-time ROI input."""
import torch
from torch import nn
import torch.nn.functional as F
from .model import RecallDenseHead


class ForegroundHead(nn.Module):
    """Independent ~77k-parameter classifier on frozen multilevel features."""
    def __init__(self):
        super().__init__()
        self.projs=nn.ModuleList([nn.Conv2d(512,32,1) for _ in range(3)])
        self.body=nn.Sequential(nn.Conv2d(96,32,3,padding=1,bias=False),
                                nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,1,1))

    def forward(self,features):
        features=features.detach()
        if features.ndim!=5 or features.shape[1:3]!=(3,512):raise ValueError('Expected B,3,512,H,W')
        x=torch.cat([p(features[:,i]) for i,p in enumerate(self.projs)],1)
        return F.interpolate(self.body(x),(224,224),mode='bilinear',align_corners=False)[:,0]


class ForegroundSeparatedModel(nn.Module):
    def __init__(self,localizer_state):
        super().__init__()
        self.localizer=RecallDenseHead()
        self.localizer.load_state_dict(localizer_state,strict=True)
        self.localizer.requires_grad_(False);self.localizer.eval()
        self.foreground=ForegroundHead()

    def train(self,mode=True):
        super().train(mode);self.localizer.eval();return self

    def forward(self,features):
        # Signature deliberately has no ROI/GT/image-intensity mask argument.
        with torch.no_grad():anomaly=self.localizer(features.detach()).sigmoid()
        logits=self.foreground(features)
        foreground=logits.sigmoid()
        return {'foreground_logits':logits,'foreground_probability':foreground,
                'anomaly_probability':anomaly,'score':anomaly*foreground}


def foreground_targets(target,margin=2):
    """Weak ROI supervises foreground only in training; all GT is foreground.

    Boundary buffer is ignored; lesion pixels always override approximate ROI.
    All uncertain pixels are excluded from the background loss.
    """
    if margin<0:raise ValueError('Invalid foreground margin')
    roi=target['roi'].bool()
    lesion=(target['hard']>0)|(target['soft']>0)
    union=roi|lesion
    if margin:
        size=2*margin+1
        dilated=F.max_pool2d(union[:,None].float(),size,1,margin)[:,0].bool()
        # Erosion with true outside-image background at image boundaries.
        padded=F.pad(roi[:,None].float(),(margin,margin,margin,margin),value=0)
        eroded=(1-F.max_pool2d(1-padded,size,1,0))[:,0]>.5
    else:dilated=union;eroded=roi
    return eroded|lesion,~dilated,lesion


def foreground_loss(logits,target,*,margin=2,lesion_weight=2.,hard_fraction=.5):
    """Each view and each region is normalized independently.

    Balanced foreground/background BCE, plus GT-lesion retention pressure.
    Half the negative term uses highest-loss background pixels.
    """
    if lesion_weight<0 or not 0<hard_fraction<=1:raise ValueError('Invalid foreground weights')
    positive,negative,lesion=foreground_targets(target,margin)
    z=logits.float()
    positive=positive.to(z.device);negative=negative.to(z.device);lesion=lesion.to(z.device)
    pos_terms=[];neg_terms=[];lesion_terms=[]
    for i in range(len(z)):
        if positive[i].any():pos_terms.append(F.softplus(-z[i][positive[i]]).mean())
        if negative[i].any():
            loss=F.softplus(z[i][negative[i]])
            k=max(1,int(loss.numel()*hard_fraction))
            neg_terms.append(.5*loss.mean()+.5*loss.topk(k).values.mean())
        if lesion[i].any():
            values=F.softplus(-z[i][lesion[i]])
            # Equal-connected-lesion weights from the existing support cache.
            weights=target['lesion_weights'][i].to(z.device)[lesion[i]].float()
            # Preserve any GT pixels outside the approximate ROI as well.
            weights=torch.where(weights>0,weights,torch.ones_like(weights))
            lesion_terms.append((values*weights).sum()/weights.sum())
    zero=z.sum()*0
    pos=torch.stack(pos_terms).mean() if pos_terms else zero
    neg=torch.stack(neg_terms).mean() if neg_terms else zero
    keep=torch.stack(lesion_terms).mean() if lesion_terms else zero
    return {'total':pos+neg+lesion_weight*keep,'foreground':pos,'background':neg,'lesion_retention':keep}
