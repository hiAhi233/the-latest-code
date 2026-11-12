"""step02a：用“最新定位头版”的真实少样本异常检测器导出 ADM 定位热图缓存。

运行环境：torch-gpu1（CUDA + 最新定位头版 + BiomedCLIP 本地缓存）。

    & $py external_tools\\export_adm_localization.py --config configs\\adm_radgenome_f0.yaml

输出（方案 §4.1）::

    <evidence.localization_dir>/
      meta.json
      index.jsonl
      arrays/<study_id>.npz     heatmap_3d[Z,heat,heat] float16
                                score_per_slice[Z] float32
                                abnormal_score scalar

正式路径只使用**预测**热图；``adm.oracle_mask: true`` 时改用 GT seg 生成
Oracle 热图并标记 ``source=ground_truth_oracle``，只能写入独立目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.config import load_config  # noqa: E402
from medrrg.data.manifest import load_manifest  # noqa: E402
from medrrg.features.adm_cache import (LOCALIZATION_ARRAYS,  # noqa: E402
                                       LOCALIZATION_FORMAT, missing_npz_keys)
from medrrg.features.adm_enhancer import (AdmParams, render_adm_grid,  # noqa: E402
                                          select_top_slices)
from medrrg.features.cache_identity import adm_localization_identity  # noqa: E402
from medrrg.utils import (now_iso, read_json, sha256_file, sha256_json,  # noqa: E402
                          write_json)

DEFAULT_DETECTOR = Path(r"D:\图神经网络\小样本原型学习\最新定位头版")
DEFAULT_CHECKPOINT = DEFAULT_DETECTOR / "runs" / "mri_foreground_k5_20261029" / "checkpoint_selected.pt"


def _source_tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resize(array: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(np.asarray(array, dtype=np.float32)).resize(
        (size[1], size[0]), mode), dtype=np.float32)


def _normalize_slice(slice_xy: np.ndarray) -> np.ndarray:
    """与定位器一致的 p1/p99 归一化，背景归零。"""
    out = np.zeros_like(slice_xy, dtype=np.float32)
    mask = slice_xy > 0
    if not mask.any():
        return out
    low, high = np.percentile(slice_xy[mask], [1.0, 99.0])
    out = np.clip((slice_xy - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _oracle_heatmap(adapter, study, data_zyx_shape, heat_size: int
                    ) -> np.ndarray | None:
    """GT seg -> [Z,heat,heat] 二值热图；缺失 seg 时返回 None。"""
    seg_path = study.meta.get("seg_path")
    if not seg_path:
        return None
    path = adapter.resolve(seg_path)
    if not Path(path).is_file():
        return None
    import nibabel as nib

    volume = np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)
    mask = np.transpose(volume > 0, (2, 0, 1)).astype(np.float32)   # [z,x,y]
    depth = mask.shape[0]
    heat = np.zeros((depth, heat_size, heat_size), dtype=np.float16)
    for z in range(depth):
        if mask[z].any():
            heat[z] = _resize(mask[z], (heat_size, heat_size), nearest=True).astype(np.float16)
    return heat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        help="Export only the requested patient partitions")
    parser.add_argument("--slice-batch", type=int, default=1,
                        help="Group axial slices before encoding their five views")
    parser.add_argument("--encoder-amp", action="store_true",
                        help="Use FP16 autocast for CUDA BiomedCLIP feature extraction")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--visualize", type=int, default=0,
                        help="为前 N 例输出 原图/热图/增强图 三联图")
    args = parser.parse_args()
    if args.slice_batch < 1:
        parser.error("--slice-batch must be positive")

    config = load_config(args.config)
    oracle = bool(config.get_path("adm.oracle_mask", False))
    chest_ct = config.get_path("localization.backend") == "chest_ct_k5"
    if chest_ct and oracle:
        raise ValueError("ChestCT expects predicted lesion heatmaps")
    source_sequence = str(config.get_path("localization.source_sequence", "t2f"))
    topk_slices = int(config.get_path("localization.topk_slices", 3))
    heat_size = int(config.get_path("localization.heat_size", 64))
    if heat_size < 8:
        raise ValueError("localization.heat_size 过小")
    adm_params = AdmParams(gamma=float(config.get_path("adm.gamma", 0.4)),
                           threshold=float(config.get_path("adm.threshold", 0.6)),
                           mode=str(config.get_path("adm.mode", "soft")),
                           normal_fallback=str(config.get_path("adm.normal_fallback",
                                                               "original")))
    output = Path(args.output) if args.output else config.path("evidence.localization_dir")
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    visualize_dir = output / "visualize"
    if args.visualize:
        visualize_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config.path("manifest"))
    from medrrg.data.adapters import build_adapter

    adapter = build_adapter(config)
    dataset_root = adapter.root
    studies = [study for study in manifest.studies
               if not args.splits or study.split in args.splits]
    if args.limit is not None:
        studies = studies[:args.limit]

    checkpoint_sha = ""
    threshold = float(config.get_path("adm.threshold", 0.6))
    if not oracle:
        detector_root = config.path("localization.detector_root", str(DEFAULT_DETECTOR))
        checkpoint_path = config.path("localization.checkpoint", str(DEFAULT_CHECKPOINT))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"定位 checkpoint 不存在：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        threshold = float(checkpoint["choice"]["threshold"])
        if chest_ct:
            from external_tools import chest_ct_locator
            from medrrg.features import ct_windows

            if checkpoint.get("format") != chest_ct_locator.FORMAT:
                raise ValueError("Expected a ChestCT K=5 checkpoint")
            if not torch.cuda.is_available():
                raise RuntimeError("ChestCT locator requires CUDA")
            if args.encoder_amp != bool(checkpoint["protocol"]["encoder_amp"]):
                raise ValueError("Use the same BiomedCLIP precision as K=5 calibration")
            torch.set_num_threads(4)
            chest_ct_locator.import_localizer(detector_root)
            from paclip.localization_v3.model import FrozenEncoder, RecallDenseHead

            encoder = FrozenEncoder().cuda().eval()
            model = RecallDenseHead().cuda().eval()
            model.load_state_dict(checkpoint["head"])
        else:
            sys.path.insert(0, str(detector_root))
            import run_localization  # noqa: F401  建立 paclip 包隔离

            from paclip.localization_v3.foreground_experiment import (
                FrozenEncoder, ForegroundSeparatedModel, extract_views, fuse_views,
                gpu_check, initialize_environment, make_views,
            )

            initialize_environment()
            gpu_check()
            if checkpoint.get("format") != "foreground_separated_v1":
                raise ValueError("不是 foreground-head checkpoint")
            c = checkpoint["config"]
            device = c["device"]
            encoder = FrozenEncoder().to(device).eval()
            model = ForegroundSeparatedModel(checkpoint["head"]).to(device)
            model.foreground.load_state_dict(checkpoint["foreground"])
            model.eval()
        checkpoint_sha = sha256_file(checkpoint_path)
        upstream_sha = _source_tree_sha(detector_root)
        preprocessing = {"source_sequence": source_sequence, "slice_size": 224,
                         "normalization": "nonzero_p1_p99", "background_zeroed": True,
                         "orientation": "closest_canonical_xy", "heat_size": heat_size,
                         "encoder_amp": args.encoder_amp, "slice_batch": args.slice_batch}
        if chest_ct:
            preprocessing.update(normalization="lung_hu_window", background_zeroed=False,
                                 orientation=ct_windows.CT_ORIENTATION,
                                 lung_hu=list(ct_windows.CT_WINDOWS["lung"]),
                                 slice_selection="global_anchors_plus_ranked",
                                 ct_window_sha256=sha256_file(Path(ct_windows.__file__)),
                                 ct_predictor_sha256=sha256_file(Path(chest_ct_locator.__file__)))
        thresholds = {"heat_threshold": threshold, "heat_size": heat_size,
                      "topk_slices": topk_slices}
        source = "predicted"
    else:
        upstream_sha = sha256_json({"oracle": "gt_mask"})
        preprocessing = {"source_sequence": source_sequence, "slice_size": 224,
                         "normalization": "gt_binary_mask", "heat_size": heat_size}
        thresholds = {"heat_threshold": 0.5, "heat_size": heat_size,
                      "topk_slices": topk_slices}
        threshold = 0.5
        source = "ground_truth_oracle"

    code_sha = sha256_json({"bridge": sha256_file(Path(__file__)),
                            "upstream": upstream_sha})

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
        invariants = {"format": LOCALIZATION_FORMAT,
                      "bridge_code_sha256": code_sha,
                      "checkpoint_sha256": checkpoint_sha,
                      "preprocessing": preprocessing, "thresholds": thresholds,
                      "source": source}
        mismatched = [key for key, value in invariants.items()
                      if previous.get(key) != value]
        if mismatched:
            raise RuntimeError(
                f"已有定位缓存与当前设置不一致：{mismatched}；"
                "请更换输出目录或显式使用 --force 重建。")

    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        array_path = output / "arrays" / f"{study.study_id}.npz"
        image = next(item for item in study.images if item.group == source_sequence)
        path = Path(image.path)
        if not path.is_absolute():
            path = dataset_root / path
        current_hash = sha256_file(path)
        current_hashes = {source_sequence: current_hash}
        if array_path.exists() and study.study_id in existing and not args.force:
            previous = dict((existing[study.study_id].get("input_sha256") or {}))
            if previous.get(source_sequence) != current_hash:
                raise RuntimeError(
                    f"{study.study_id}: 输入影像已变化；请使用 --force 重建定位缓存。")
            missing = missing_npz_keys(array_path, LOCALIZATION_ARRAYS)
            if missing:
                raise RuntimeError(
                    f"{study.study_id}: 定位缓存缺少字段 {missing}；"
                    "请使用 --force 重建定位缓存。")
            print(f"[skip] {study.study_id}", flush=True)
            image_hashes[study.study_id] = current_hashes
            rows.append(existing[study.study_id])
            continue

        import nibabel as nib

        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = np.asanyarray(nii.dataobj, dtype=np.float32)     # [X,Y,Z]
        if data.ndim != 3:
            raise ValueError(f"{study.study_id}: 期望 3D，收到 {data.shape}")
        height, width, depth = data.shape
        scores = np.zeros(depth, dtype=np.float32)
        if oracle:
            heat = _oracle_heatmap(adapter, study, (depth, height, width), heat_size)
            if heat is None:
                raise FileNotFoundError(
                    f"{study.study_id}: Oracle 需要 seg，但 seg_path 不存在")
            scores = np.asarray(heat, dtype=np.float32).mean(axis=(1, 2))
            # GT 是 Oracle 证据，不存在“异常置信度不足”的情况；固定 1.0，
            # 避免下游误按预测置信度回退原图（O0 上限实验失真）。
            abnormal = 1.0
        elif chest_ct:
            heat = chest_ct_locator.predict_volume(data, encoder, model, args.slice_batch,
                                                   args.encoder_amp, heat_size)
            scores[:] = [chest_ct_locator.hottest_mean(row, .01) for row in heat]
            abnormal = chest_ct_locator.hottest_mean(heat, .001)
        else:
            heat = np.zeros((depth, heat_size, heat_size), dtype=np.float16)
            active = [z for z in range(depth) if np.any(data[:, :, z])]
            for start in range(0, len(active), args.slice_batch):
                zs = active[start:start + args.slice_batch]
                image_views = []
                for z in zs:
                    gray = _resize(_normalize_slice(data[:, :, z]), (224, 224))
                    empty = np.zeros_like(gray)
                    ones = np.ones_like(gray, dtype=bool)
                    views, _ = make_views(gray, empty, empty, ones, crop=c["crop_size"])
                    image_views.append(views)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=args.encoder_amp):
                    features = extract_views(encoder, torch.cat(image_views), device)
                features = features.to(device, dtype=torch.float32)
                with torch.no_grad():
                    view_maps = model(features)["score"].split(image_views[0].shape[0])
                    maps = torch.stack([fuse_views(values, c) for values in view_maps]).cpu().numpy()
                for z, score in zip(zs, maps):
                    heat_slice = _resize(score, (heat_size, heat_size)).astype(np.float16)
                    heat[z] = heat_slice
                    flat_slice = np.sort(heat_slice.astype(np.float32).reshape(-1))
                    top = max(1, flat_slice.size // 100)
                    scores[z] = float(flat_slice[-top:].mean())
            flat = np.sort(heat.reshape(-1).astype(np.float32))
            abnormal = float(flat[-max(1, flat.size // 1000):].mean()) if flat.size else 0.0

        selected_z = (ct_windows.select_ct_slices(scores, topk_slices) if chest_ct
                      else select_top_slices(scores, topk_slices))
        np.savez_compressed(array_path, heatmap_3d=heat,
                            score_per_slice=scores.astype(np.float32),
                            abnormal_score=np.float32(abnormal))
        image_hashes[study.study_id] = current_hashes
        rows.append({
            "study_id": study.study_id,
            "source_sequence": source_sequence,
            "array_path": f"arrays/{study.study_id}.npz",
            "shape_zyx": [depth, width, height] if chest_ct else [depth, height, width],
            "heat_size": int(heat_size),
            "threshold": float(threshold),
            "selected_z": [int(z) for z in selected_z],
            "abnormal_score": float(abnormal),
            "source": source,
            "checkpoint_sha256": checkpoint_sha,
            "input_sha256": current_hashes,
        })
        print(f"[localization] {index + 1}/{len(studies)} {study.study_id} "
              f"z={selected_z} abnormal={abnormal:.4f}", flush=True)

        if args.visualize and index < args.visualize:
            volume = None if chest_ct else _normalize_slices(data)
            for z in selected_z:
                original = (ct_windows.ct_window(data[:, :, z], "lung") if chest_ct
                            else volume[:, :, z])
                enhanced, meta = _enhance_preview(original, np.asarray(heat[z]),
                                                  adm_params, abnormal, oracle)
                render_adm_grid(original, np.asarray(heat[z]), enhanced, adm_params,
                                abnormal).save(
                    visualize_dir / f"{study.study_id}_z{z:03d}.png")
                (visualize_dir / f"{study.study_id}_z{z:03d}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(output / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    identity = adm_localization_identity(code_sha, checkpoint_sha, preprocessing,
                                         thresholds, image_hashes)
    meta = {
        "format": LOCALIZATION_FORMAT,
        "identity": identity,
        "source": source,
        "checkpoint_sha256": checkpoint_sha,
        "detector_code_sha256": upstream_sha,
        "bridge_code_sha256": code_sha,
        "preprocessing": preprocessing,
        "thresholds": thresholds,
        "studies": sorted(row["study_id"] for row in rows),
        "created_at": now_iso(),
        "notes": "只保存预测热图与逐片异常分数；预测框不再进入报告模型。"
                 "Oracle 使用 GT seg，只能写入独立目录。",
    }
    write_json(output / "meta.json", meta)
    print(f"[localization] 写入 {len(rows)} 例 -> {output}", flush=True)
    return 0


def _normalize_slices(data_xyz: np.ndarray) -> np.ndarray:
    """整卷 p1/p99 归一化，返回 [X,Y,Z] 0..1。"""
    out = np.zeros_like(data_xyz, dtype=np.float32)
    mask = data_xyz > 0
    if not mask.any():
        return out
    low, high = np.percentile(data_xyz[mask], [1.0, 99.0])
    out = np.clip((data_xyz - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _enhance_preview(original: np.ndarray, heat: np.ndarray, params: AdmParams,
                     abnormal: float, oracle: bool) -> tuple[np.ndarray, dict]:
    from medrrg.features.adm_enhancer import enhance_image

    return enhance_image(original, heat, params, abnormal_score=abnormal,
                         oracle_mask=oracle)


if __name__ == "__main__":
    raise SystemExit(main())
