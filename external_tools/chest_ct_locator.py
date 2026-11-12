"""BiomedCLIP + RecallDenseHead inference used for calibration and CT export."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from medrrg.features.ct_windows import ct_window


CROP_SIZE = 144
ZOOM_WEIGHT = 0.5
HEAT_SIZE = 64
FORMAT = "chest_ct_recall_k5_v1"


def import_localizer(detector_root: Path):
    sys.path.insert(0, str(detector_root))
    import run_localization

    if Path(run_localization.__file__).resolve().parent != detector_root.resolve():
        raise RuntimeError("Unexpected PA-CLIP source")


def resize(array: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.float32))
                      .resize((size, size), Image.Resampling.BILINEAR))


def hottest_mean(heat: np.ndarray, fraction: float) -> float:
    values = heat.astype(np.float32).ravel()
    first = values.size - max(1, int(values.size * fraction))
    return float(np.partition(values, first)[first:].mean())


@torch.inference_mode()
def predict_volume(volume: np.ndarray, encoder, head, slice_batch: int,
                   amp: bool, heat_size: int = HEAT_SIZE) -> np.ndarray:
    from paclip.localization_v3.data import MEAN, STD, extract_views
    from paclip.localization_v3.model import combine_views, view_boxes

    boxes = view_boxes(crop=CROP_SIZE)
    device = next(encoder.parameters()).device
    heat = np.empty((volume.shape[2], heat_size, heat_size), dtype=np.float16)
    for start in range(0, volume.shape[2], slice_batch):
        stop = min(start + slice_batch, volume.shape[2])
        slices = np.stack([resize(ct_window(volume[:, :, z], "lung"), 224)
                           for z in range(start, stop)])
        axial = torch.from_numpy(slices)[:, None]
        views = torch.stack([
            F.interpolate(axial[:, :, y0:y1, x0:x1], (224, 224),
                          mode="bilinear", align_corners=False)
            for y0, x0, y1, x1 in boxes], dim=1).flatten(0, 1)
        views = (views - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            features = extract_views(encoder, views, device, batch_size=32)
        probabilities = head(features.to(device, dtype=torch.float32)).sigmoid()
        fused = []
        for maps in probabilities.split(len(boxes)):
            full, zoom = combine_views(maps, boxes)
            fused.append((1 - ZOOM_WEIGHT) * full + ZOOM_WEIGHT * zoom)
        heat[start:stop] = F.interpolate(torch.stack(fused)[:, None], (heat_size, heat_size),
                                         mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    return heat
