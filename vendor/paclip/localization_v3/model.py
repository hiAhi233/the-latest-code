import torch
from torch import nn
import torch.nn.functional as F
from paclip.config import Config
from paclip.models.paclip import TextSideAnomalyModel


class FrozenEncoder(nn.Module):
    """Use the original checkout's BiomedCLIP; no learned adapter initialization."""
    def __init__(self):
        super().__init__()
        self.base = TextSideAnomalyModel(Config())
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        _, _, grid, levels = self.base.encode_image(x, return_levels=True)
        if levels.shape[1] != 3 or levels.shape[-1] != 512:
            raise ValueError('Expected three independent 512-dimensional levels')
        return levels.permute(0, 1, 3, 2).reshape(len(x), 3, 512, *grid)


class DenseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1, bias=False), nn.GELU())
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # 1x1 and bilinear interpolation commute; using the small grid saves RAM.
        z = self.out(self.body(x))
        return F.interpolate(z, (224, 224), mode='bilinear', align_corners=False)


class RecallDenseHead(nn.Module):
    """Same spatial decoder capacity in reference and new-loss experiments."""
    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(512, 128, 1) for _ in range(3)])
        self.decoder = DenseDecoder()

    def forward(self, levels):
        if levels.ndim != 5 or levels.shape[1:3] != (3, 512):
            raise ValueError('Expected B,3,512,H,W')
        return self.decoder(torch.cat([p(levels[:, i]) for i, p in enumerate(self.projs)], 1))[:, 0]


def view_boxes(size=224, crop=144):
    """Fixed overlapping coverage, independent of GT and prediction confidence."""
    if not 0 < crop < size:
        raise ValueError('Invalid crop size')
    return [(0, 0, size, size)] + [
        (y, x, y + crop, x + crop) for y in (0, size - crop) for x in (0, size - crop)]


def combine_views(probabilities, boxes):
    """Return global and overlap-averaged zoom maps, with no thresholding."""
    if len(probabilities) != len(boxes) or len(boxes) < 2:
        raise ValueError('Expected global plus zoom views')
    size = boxes[0][2]
    result = probabilities.new_zeros((size, size))
    count = probabilities.new_zeros((size, size))
    for p, (y0, x0, y1, x1) in zip(probabilities[1:], boxes[1:]):
        v = F.interpolate(p[None, None], (y1-y0, x1-x0), mode='bilinear', align_corners=False)[0, 0]
        # Positive taper avoids uncovered image edges and softens crop seams.
        wy = torch.hann_window(y1-y0, periodic=False, device=p.device).clamp_min(.1)
        wx = torch.hann_window(x1-x0, periodic=False, device=p.device).clamp_min(.1)
        weight = wy[:, None] * wx[None, :]
        result[y0:y1, x0:x1] += v * weight
        count[y0:y1, x0:x1] += weight
    if (count == 0).any():
        raise ValueError('Zoom windows do not cover the image')
    return probabilities[0], result / count
