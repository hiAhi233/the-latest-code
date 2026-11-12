from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label
import torch
import torch.nn.functional as F
from .model import view_boxes

MEAN = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
STD = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def decode_gray(image):
    """Match the reference loader for both uint8 MRI and float [0,1] CT NPZs."""
    image=np.asarray(image)
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite source image')
    if image.dtype == np.uint8:
        u8=image
    elif np.issubdtype(image.dtype,np.floating) and image.min()>=0 and image.max()<=1+1e-6:
        # The supplied CT reference truncates to uint8 before CLIP normalization.
        u8=np.clip(image*255.,0,255).astype(np.uint8)
    else:
        raise ValueError('Expected uint8 or floating [0,1] image')
    return u8.astype(np.float32)/255.


class SliceSet:
    """Read explicitly split NPZs; train is exactly the supplied support list.

    Expected fields: image (uint8), mask224, roi224. Training may additionally
    supply source_mask to preserve native soft-area supervision.
    """
    def __init__(self, manifest, support):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        self.rows = self.manifest['index']
        ids = [r['file'] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate sample identities across splits')
        entries = json.loads(Path(support).read_text(encoding='utf-8'))['samples']
        wanted = {r['file'] for r in entries}
        if len(wanted) != len(entries):
            raise ValueError('Repeated support sample')
        self.splits = {s: [r for r in self.rows if r['split'] == s] for s in ('train', 'val', 'test')}
        self.splits['train'] = [r for r in self.splits['train'] if r['file'] in wanted]
        if {r['file'] for r in self.splits['train']} != wanted:
            raise ValueError('Support includes a non-training or missing sample')
        if {r['label'] for r in self.splits['train']} != {0, 1}:
            raise ValueError('Both normal and abnormal support required')
        if not self.splits['val'] or not self.splits['test']:
            raise ValueError('Separate validation and test sets required')
        if any(Path(r['file']).name != r['file'] for r in self.rows):
            raise ValueError('Use unique NPZ basenames in manifest')
        self.identity = {'manifest_sha256': sha(manifest), 'support_sha256': sha(support),
                         'samples': {r['file']: sha(self.root/r['file']) for r in self.rows}}
        self.identity['native_training_masks'] = {
            r['file']: sha(r['source_mask']) for r in self.splits['train'] if r.get('source_mask')}
        self.patient_verified = bool(self.manifest.get('patient_disjoint_verified', False))
        if self.patient_verified:
            cases = {s: {r.get('case') for r in rs} for s, rs in self.splits.items()}
            if any(None in cs for cs in cases.values()) or any(
                    cases[a] & cases[b] for a, b in [('train','val'),('train','test'),('val','test')]):
                raise ValueError('Claimed patient independence contradicts manifest')

    def read(self, row, native=False):
        with np.load(self.root/row['file'], allow_pickle=False) as d:
            gray = decode_gray(d['image'])
            hard = np.asarray(d['mask224']) > 0
            roi = np.asarray(d['roi224']) > 0
        if gray.shape != (224, 224) or hard.shape != gray.shape or roi.shape != gray.shape:
            raise ValueError('Expected aligned 224 grids')
        if not np.isfinite(gray).all() or not roi.any() or bool(hard.any()) != bool(row['label']):
            raise ValueError('Invalid image/ROI or inconsistent mask and label')
        soft = hard.astype(np.float32)
        if native and row.get('source_mask'):
            with Image.open(row['source_mask']) as im:
                mask = torch.from_numpy((np.asarray(im.convert('L')) > 0).astype(np.float32))
            soft = F.adaptive_avg_pool2d(mask[None, None], (224,224))[0,0].numpy()
            preserved = F.adaptive_max_pool2d(mask[None, None], (224,224))[0,0].numpy() > 0
            if not np.array_equal(preserved, hard):
                raise ValueError('Native mask disagrees with exported GT')
        return gray, soft, hard, roi


def make_views(gray, soft, hard, roi, flip=False, crop=144):
    # Flip the actual image before encoding; feature-grid flipping is not used.
    arrays = [np.ascontiguousarray(a[:, ::-1] if flip else a) for a in (gray, soft, hard, roi)]
    images, targets = [], []
    for y0,x0,y1,x1 in view_boxes(crop=crop):
        g,s,h,r = [torch.from_numpy(a[y0:y1,x0:x1].astype(np.float32))[None,None] for a in arrays]
        g = F.interpolate(g, (224,224), mode='bilinear', align_corners=False)[0,0]
        s = F.interpolate(s, (224,224), mode='bilinear', align_corners=False)[0,0]
        h = F.interpolate(h, (224,224), mode='nearest')[0,0]
        r = F.interpolate(r, (224,224), mode='nearest')[0,0].bool()
        components, n = label(h.numpy() > 0, np.ones((3,3)))
        weights = np.zeros((224,224), np.float32)
        for i in range(1, n+1):
            region = (components == i) & r.numpy()
            if region.any(): weights[region] = 1.0 / int(region.sum())
        images.append((g.expand(3,-1,-1)-MEAN)/STD)
        targets.append({'soft':s, 'hard':h, 'roi':r, 'lesion_weights':torch.from_numpy(weights)})
    return torch.stack(images), targets


@torch.no_grad()
def extract_views(encoder, images, device, batch_size=8):
    return torch.cat([encoder(x.to(device)).cpu() for x in images.split(batch_size)])
