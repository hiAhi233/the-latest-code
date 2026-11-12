"""HU preprocessing shared by the chest CT locator and feature exporters."""
import numpy as np


CT_WINDOWS = {"lung": (-1000, 400), "mediastinal": (-160, 240)}
CT_ORIENTATION = "canonical_ras_axial_radiological"


def radiological_axial(slice_xy: np.ndarray) -> np.ndarray:
    """Display canonical RAS axial data with anterior up and patient right left."""
    return np.ascontiguousarray(slice_xy.T[::-1, ::-1])


def ct_window(slice_xy: np.ndarray, window: str) -> np.ndarray:
    low, high = CT_WINDOWS[window]
    hu = radiological_axial(slice_xy)
    return np.clip((hu - low) / (high - low), 0, 1).astype(np.float32)


def select_ct_slices(scores: np.ndarray, count: int) -> list[int]:
    """Retain fixed global views and fill remaining slots by heatmap rank."""
    if not 1 <= count <= len(scores):
        raise ValueError("The requested CT slice count exceeds the volume depth")
    anchors = np.linspace(0.15, 0.85, max(1, count // 2)) * (len(scores) - 1)
    selected = set(anchors.round().astype(int).tolist())
    rank = np.argsort(-scores, kind="stable")
    for gap in (3, 1):
        for z in rank:
            if len(selected) == count:
                return sorted(selected)
            if all(abs(int(z) - other) >= gap for other in selected):
                selected.add(int(z))
    return sorted(selected)
