"""Exact, memory-bounded pixel AUROC/AP from saved continuous maps.

Only positive scores are sorted. Negative scores are counted relative to each
unique positive threshold; this preserves exact ties without score binning.
"""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .metrics import operating_metrics


class OperatingAccumulator:
    def __init__(self,threshold):
        self.threshold=float(threshold)
        self.pixels=dict(tp=0,fp=0,fn=0)
        self.counts=dict(lesions=0,detected=0,small_lesions=0,small_detected=0)
        self.positive_dice=[];self.normal_fp=[]

    def add(self,score,gt,roi):
        m=operating_metrics([score],[gt],[roi],self.threshold)
        for key in self.pixels:self.pixels[key]+=m['pixels'][key]
        for key in self.counts:self.counts[key]+=m['counts'][key]
        if m['positive_macro_dice'] is not None:self.positive_dice.append(m['positive_macro_dice'])
        if m['normal_fp_area'] is not None:self.normal_fp.append(m['normal_fp_area'])

    def result(self):
        p=self.pixels;c=self.counts
        return {'dice_micro':2*p['tp']/max(2*p['tp']+p['fp']+p['fn'],1),
                'positive_macro_dice':float(np.mean(self.positive_dice)) if self.positive_dice else None,
                'lesion_recall':c['detected']/c['lesions'] if c['lesions'] else None,
                'small_lesion_recall':c['small_detected']/c['small_lesions'] if c['small_lesions'] else None,
                'normal_fp_area':float(np.mean(self.normal_fp)) if self.normal_fp else None,
                'counts':dict(c),'pixels':dict(p),'threshold':self.threshold}


def binary_auc(labels,scores):
    labels=np.asarray(labels,bool);positive=int(labels.sum());negative=len(labels)-positive
    if not positive or not negative:return None
    ranks=rankdata(scores,method='average')
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def exact_rank_from_factory(factory,negative_chunk_size=2_000_000):
    """factory must yield (continuous score array, binary GT) identically twice."""
    positives=[];labels=[];image_scores=[];negative_count=0
    for score,gt in factory():
        score=np.asarray(score);gt=np.asarray(gt,bool)
        if score.shape!=gt.shape or not np.isfinite(score).all():raise ValueError('Invalid score/GT arrays')
        positives.append(score[gt]);negative_count+=int((~gt).sum())
        labels.append(bool(gt.any()));image_scores.append(float(score.max()))
    positive=np.concatenate(positives)
    positive_count=int(positive.size)
    image_auc=binary_auc(labels,image_scores)
    if not positive_count or not negative_count:
        return {'image_auroc':image_auc,'pixel_auroc':None,'pixel_ap':1. if positive_count else 0.}
    values,counts=np.unique(positive,return_counts=True)
    del positive,positives
    size=len(values);negative_bins=np.zeros(size+1,dtype=np.int64);equal=np.zeros(size,dtype=np.int64)
    parts=[];buffered=0;second_negative_count=0
    def consume(block):
        indices=np.searchsorted(values,block,side='left')
        negative_bins[:]+=np.bincount(indices,minlength=size+1)
        valid=indices<size
        locations=indices[valid];raw=block[valid]
        same=locations[values[locations]==raw]
        if len(same):equal[:]+=np.bincount(same,minlength=size)
    for score,gt in factory():
        negative=np.asarray(score)[~np.asarray(gt,bool)]
        second_negative_count+=len(negative)
        parts.append(negative);buffered+=len(negative)
        if buffered>=negative_chunk_size:
            consume(np.concatenate(parts));parts=[];buffered=0
    if parts:consume(np.concatenate(parts))
    if second_negative_count!=negative_count:raise ValueError('Metric factory changed between passes')
    less=np.cumsum(negative_bins[:-1],dtype=np.int64)-equal
    concordant=float(np.sum(counts.astype(np.float64)*(less+.5*equal)))
    true_positive=np.cumsum(counts[::-1],dtype=np.int64)[::-1]
    selected_negative=negative_count-less
    ap=float(np.sum((counts.astype(np.float64)/positive_count)*true_positive/(true_positive+selected_negative)))
    return {'image_auroc':image_auc,'pixel_auroc':concordant/(positive_count*negative_count),'pixel_ap':ap}


def rank_npz(files,score_key,negative_chunk_size=2_000_000):
    files=[Path(p) for p in files]
    def factory():
        for file in files:
            with np.load(file) as d:yield d[score_key],d['gt'].astype(bool)
    return exact_rank_from_factory(factory,negative_chunk_size)
