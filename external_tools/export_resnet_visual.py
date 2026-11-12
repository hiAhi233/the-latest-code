"""step02b：ADM 增强 + 冻结 ResNet-101 导出视觉 patch tokens 缓存。

运行环境：torch-gpu1（CUDA；ResNet-101 ImageNet 权重走本机 torchvision 缓存）。

    & $py external_tools\\export_resnet_visual.py --config configs\\adm_radgenome_f0.yaml

输出（方案 §4.2、§4.3）::

    <evidence.visual_dir>/
      meta.json  index.jsonl
      visual/<study_id>.npz
        visual_tokens       [108,256] float32   ADM 增强图 tokens（B0 时为原图）
        visual_tokens_null  [108,256] float32   同一 ResNet 对原图的 tokens（null-adm）
        visual_mask         [108] bool
        visual_modality     [108] int64  0..3 = T1/T1C/T2/FLAIR
        visual_z            [108] float32 归一化 z
        visual_view         [108] int64
        selected_z          [3] int64
        adm_applied         scalar bool

token 顺序：``for z in selected_z: for modality in [t1n,t1c,t2w,t2f]: 9 patches``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.config import load_config  # noqa: E402
from medrrg.features import ct_windows  # noqa: E402
from medrrg.data.manifest import load_manifest  # noqa: E402
from medrrg.features.adm_cache import (AdmLocalizationCache,  # noqa: E402
                                       VISUAL_ARRAYS, VISUAL_FORMAT,
                                       missing_npz_keys)
from medrrg.features.adm_enhancer import (AdmParams, enhance_image,  # noqa: E402
                                          render_adm_grid, resize_heatmap)
from medrrg.features.cache_identity import resnet_visual_identity  # noqa: E402
from medrrg.features.resnet_visual import (ResNetPatchEncoder, encoder_summary,  # noqa: E402
                                           slice_to_tensor)
from medrrg.utils import (now_iso, read_json, sha256_file, sha256_json,  # noqa: E402
                          write_json)

DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def validate_existing_meta(previous: dict, invariants: dict) -> None:
    """复用已有视觉缓存前，核对格式版本与全部生成参数。

    旧版（v1）缓存没有 patch 位置字段：如果只比较坐标/权重而漏掉格式，脚本会
    “跳过重建”却在末尾写入 v2 meta，训练时才发现缺字段。这里把 ``format`` 作为
    硬条件，格式不一致直接要求 ``--force`` 重建。
    """
    mismatched = [key for key, value in invariants.items()
                  if previous.get(key) != value]
    if not mismatched:
        return
    hint = ""
    if previous.get("format") != invariants.get("format"):
        hint = (f"\n缓存格式 {previous.get('format')!r} 与当前 "
                f"{invariants.get('format')!r} 不一致（v2 起需要 visual_patch_row/col）。")
    raise RuntimeError(
        f"已有视觉缓存与当前设置不一致：{mismatched}{hint}\n"
        "请更换输出目录或显式使用 --force 重建。")


def validate_existing_file(path: Path, study_id: str) -> None:
    """复用单个 NPZ 前检查 v2 必需字段，避免半旧文件混入新缓存。"""
    missing = missing_npz_keys(path, VISUAL_ARRAYS)
    if missing:
        raise RuntimeError(
            f"{study_id}: 视觉缓存缺少字段 {missing}；"
            "请使用 --force 重建该缓存目录。")


def _normalize_slice(slice_xy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(slice_xy, dtype=np.float32)
    mask = slice_xy > 0
    if not mask.any():
        return out
    low, high = np.percentile(slice_xy[mask], [1.0, 99.0])
    out = np.clip((slice_xy - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _load_volume(adapter, study, group: str) -> np.ndarray:
    import nibabel as nib

    image = next(item for item in study.images if item.group == group)
    path = Path(image.path)
    if not path.is_absolute():
        path = adapter.root / path
    volume = np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj,
                           dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"{study.study_id}/{group}: 期望 3D，收到 {volume.shape}")
    return volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--visualize", type=int, default=0,
                        help="为前 N 例每个选中切片输出 原图/热图/增强图 三联图")
    args = parser.parse_args()

    config = load_config(args.config)
    chest_ct = config.get_path("dataset.modality") == "CT"
    adm_enabled = bool(config.get_path("adm.enabled", True))
    oracle = bool(config.get_path("adm.oracle_mask", False))
    params = AdmParams(gamma=float(config.get_path("adm.gamma", 0.4)),
                       threshold=float(config.get_path("adm.threshold", 0.6)),
                       mode=str(config.get_path("adm.mode", "soft")),
                       normal_fallback=str(config.get_path("adm.normal_fallback",
                                                           "original")))
    modalities = [str(item) for item in config.get_path("visual.modalities",
                                                        DEFAULT_MODALITIES)]
    patch_grid = int(config.get_path("visual.patch_grid", 3))
    output_dim = int(config.get_path("visual.output_dim", 256))
    image_size = int(config.get_path("visual.image_size", 224))
    pretrained = bool(config.get_path("visual.pretrained", "imagenet") == "imagenet")
    frozen = bool(config.get_path("visual.frozen", True))
    topk_slices = int(config.get_path("localization.topk_slices", 3))

    output = Path(args.output) if args.output else config.path("evidence.visual_dir")
    (output / "visual").mkdir(parents=True, exist_ok=True)
    visualize_dir = output / "visualize"
    if args.visualize:
        visualize_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config.path("manifest"))
    from medrrg.data.adapters import build_adapter

    adapter = build_adapter(config)
    studies = manifest.studies[: int(args.limit)] if args.limit else manifest.studies
    localization = AdmLocalizationCache.load(config.path("evidence.localization_dir"))
    localization.verify([study.study_id for study in studies],
                        require_predicted=not oracle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_seed = int(config.get_path("visual.proj_seed", 0))
    encoder = ResNetPatchEncoder(backbone="resnet101", pretrained=pretrained,
                                 patch_grid=patch_grid, output_dim=output_dim,
                                 frozen=frozen, proj_seed=proj_seed).to(device)
    summary = encoder_summary(encoder)
    resnet_sha = summary["weights_sha256"]
    projection_sha = summary["projection_sha256"]
    tokens_per_slice = encoder.num_patches

    adm_meta = {"enabled": adm_enabled, **params.as_dict(), "oracle_mask": oracle,
                "image_size": image_size, "normalization": "nonzero_p1_p99"}
    if chest_ct:
        adm_meta.update(normalization="ct_hu_window", orientation=ct_windows.CT_ORIENTATION,
                        windows_hu={g: list(ct_windows.CT_WINDOWS[g]) for g in modalities},
                        ct_window_sha256=sha256_file(Path(ct_windows.__file__)))
    sampling = {"topk_slices": topk_slices,
                "source_sequence": str(config.get_path("localization.source_sequence",
                                                      "t2f"))}
    visual_meta = {"backbone": "resnet101", "patch_grid": patch_grid,
                   "output_dim": output_dim, "image_size": image_size,
                   "modalities": modalities, "frozen": frozen,
                   "tokens_per_slice": tokens_per_slice,
                   "proj_seed": proj_seed, "projection_sha256": projection_sha}

    rows: list[dict] = []
    existing: dict[str, dict] = {}
    index_path = output / "index.jsonl"
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    existing[str(row["study_id"])] = row
    meta_path = output / "meta.json"
    if meta_path.is_file() and not args.force:
        previous = read_json(meta_path)
        invariants = {"format": VISUAL_FORMAT,
                      "localization_identity": localization.identity,
                      "resnet_weights_sha256": resnet_sha, "adm": adm_meta,
                      "sampling": sampling, "visual": visual_meta,
                      "d_visual": output_dim, "tokens_per_study": tokens_per_slice
                      * len(modalities) * topk_slices}
        validate_existing_meta(previous, invariants)

    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        visual_path = output / "visual" / f"{study.study_id}.npz"
        location = localization.get(study.study_id)
        paths: dict[str, Path] = {}
        current_hashes: dict[str, str] = {}
        for group in modalities:
            image = next((item for item in study.images if item.group == group), None)
            if image is None:
                raise ValueError(f"{study.study_id}: 缺少序列 {group}")
            path = Path(image.path)
            if not path.is_absolute():
                path = adapter.root / path
            paths[group] = path
            current_hashes[group] = sha256_file(path)
        image_hashes[study.study_id] = current_hashes
        if visual_path.exists() and study.study_id in existing and not args.force:
            previous = dict((existing[study.study_id].get("input_sha256") or {}))
            if previous != current_hashes:
                raise RuntimeError(
                    f"{study.study_id}: 输入影像已变化；请使用 --force 重建视觉缓存。")
            validate_existing_file(visual_path, study.study_id)
            print(f"[skip] {study.study_id}", flush=True)
            rows.append(existing[study.study_id])
            continue

        selected_z = [int(z) for z in location.selected_z[:topk_slices]]
        if not selected_z:
            raise ValueError(f"{study.study_id}: 定位缓存没有 selected_z")
        shape_z = int(location.shape_zyx[0])
        if chest_ct and len(set(paths.values())) == 1:
            volume = _load_volume(adapter, study, modalities[0])
            volumes = dict.fromkeys(modalities, volume)
        else:
            volumes = {group: _load_volume(adapter, study, group) for group in modalities}

        slices: list[np.ndarray] = []
        null_slices: list[np.ndarray] = []
        modality_ids: list[int] = []
        z_positions: list[float] = []
        view_ids: list[int] = []
        patch_rows: list[int] = []
        patch_cols: list[int] = []
        # patch 顺序 = AdaptiveAvgPool2d(patch_grid) 的行优先展平
        slice_patch_rows = [p // patch_grid for p in range(tokens_per_slice)]
        slice_patch_cols = [p % patch_grid for p in range(tokens_per_slice)]
        fallbacks = 0
        preview: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for z in selected_z:
            heat = location.heatmap_at(z)
            for modality_index, group in enumerate(modalities):
                volume = volumes[group]
                if not 0 <= z < volume.shape[2]:
                    raise ValueError(f"{study.study_id}: z={z} 超出 {volume.shape[2]}")
                axial = volume[:, :, z]
                original = ct_windows.ct_window(axial, group) if chest_ct else _normalize_slice(axial)
                if original.shape != (image_size, image_size):
                    # 与定位器一致：先在 224×224 上计算热图，ADM 也在同一尺度增强
                    original = resize_heatmap(original, (image_size, image_size))
                if adm_enabled:
                    enhanced, meta = enhance_image(
                        original, heat, params,
                        abnormal_score=location.abnormal_score,
                        oracle_mask=oracle)
                    fallbacks += int(bool(meta.get("fallback")))
                else:
                    enhanced, meta = original.copy(), {"fallback": False,
                                                       "reason": "adm_disabled"}
                slices.append(enhanced)
                null_slices.append(original)
                modality_ids.extend([modality_index] * tokens_per_slice)
                z_norm = float(z) / max(1, shape_z - 1)
                z_positions.extend([z_norm] * tokens_per_slice)
                view_ids.extend([0] * tokens_per_slice)
                patch_rows.extend(slice_patch_rows)
                patch_cols.extend(slice_patch_cols)
                if args.visualize and index < args.visualize and len(preview) < 6:
                    preview.append((z, modality_index, original, heat, enhanced))

        batch = torch.stack([slice_to_tensor(item) for item in slices]).to(device)
        batch_null = torch.stack([slice_to_tensor(item) for item in null_slices]).to(device)
        with torch.no_grad():
            tokens = encoder(batch).cpu().numpy().astype(np.float32)
            tokens_null = encoder(batch_null).cpu().numpy().astype(np.float32)
        tokens = tokens.reshape(-1, output_dim)
        tokens_null = tokens_null.reshape(-1, output_dim)
        count = tokens.shape[0]
        expected = len(selected_z) * len(modalities) * tokens_per_slice
        if count != expected:
            raise RuntimeError(f"{study.study_id}: token 数 {count} != 期望 {expected}")
        visual_mask = np.ones(count, dtype=bool)
        np.savez_compressed(
            visual_path,
            visual_tokens=tokens,
            visual_tokens_null=tokens_null,
            visual_mask=visual_mask,
            visual_modality=np.asarray(modality_ids, dtype=np.int64),
            visual_z=np.asarray(z_positions, dtype=np.float32),
            visual_view=np.asarray(view_ids, dtype=np.int64),
            visual_patch_row=np.asarray(patch_rows, dtype=np.int64),
            visual_patch_col=np.asarray(patch_cols, dtype=np.int64),
            selected_z=np.asarray(selected_z, dtype=np.int64),
            adm_applied=np.asarray(bool(adm_enabled)))
        rows.append({
            "study_id": study.study_id,
            "array_path": f"visual/{study.study_id}.npz",
            "num_tokens": int(count),
            "d_visual": int(output_dim),
            "selected_z": [int(z) for z in selected_z],
            "adm_applied": bool(adm_enabled),
            "num_fallbacks": int(fallbacks),
            "input_sha256": current_hashes,
        })
        print(f"[resnet] {index + 1}/{len(studies)} {study.study_id} "
              f"tokens={count} z={selected_z} fallback={fallbacks}", flush=True)

        if preview:
            for z, modality_index, original, heat, enhanced in preview:
                render_adm_grid(original, heat, enhanced, params,
                                location.abnormal_score).save(
                    visualize_dir
                    / f"{study.study_id}_z{z:03d}_{modalities[modality_index]}.png")

    with open(output / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    identity = resnet_visual_identity(
        localization.identity, resnet_sha, adm_meta, sampling, visual_meta, image_hashes)
    meta = {
        "format": VISUAL_FORMAT,
        "identity": identity,
        "localization_identity": localization.identity,
        "resnet_weights_sha256": resnet_sha,
        "resnet_source": "torchvision resnet101 IMAGENET1K_V1",
        "adm": adm_meta,
        "sampling": sampling,
        "visual": visual_meta,
        "d_visual": int(output_dim),
        "tokens_per_study": int(tokens_per_slice * len(modalities) * topk_slices),
        "studies": sorted(row["study_id"] for row in rows),
        "image_sha256": image_hashes,
        "created_at": now_iso(),
        "notes": "每例 3 个异常切片 × 4 序列 × 9 patch = 108 tokens；"
                 "visual_tokens_null 为同一 ResNet 对原图的编码，用于 null-adm 干预。",
    }
    write_json(output / "meta.json", meta)
    print(f"[resnet] 写入 {len(rows)} 例 -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
