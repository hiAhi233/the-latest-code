"""step02c：冻结 UniMed-CLIP 零样本疾病语义导出（13 概念）。

运行环境：torch-gpu1（UniMed-CLIP 依赖 + 2.35GB 权重 + BiomedBERT 文本塔）。

    & $py external_tools\\export_unimed_semantic.py --config configs\\adm_radgenome_f0.yaml

输入使用与视觉支路**相同的代表性原图**（不做 ADM，避免改变预训练分布）。

输出（方案 §4.4）::

    <evidence.semantic_dir>/
      meta.json  index.jsonl  prompts.json
      semantic/<study_id>.npz
        concept_ids       [13]  str
        text_prototypes   [13,512] float32（L2 归一化）
        probabilities     [13]  float32（sigmoid(scale×margin)，未校准）
        confidence        [13]  float32
        margins           [13]  float32

语义不是 hash/probe，也不是诊断概率。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from medrrg.config import load_config  # noqa: E402
from medrrg.features import ct_windows  # noqa: E402
from medrrg.data.manifest import load_manifest  # noqa: E402
from medrrg.features.adm_cache import (AdmLocalizationCache,  # noqa: E402
                                       SEMANTIC_ARRAYS, SEMANTIC_FORMAT,
                                       missing_npz_keys)
from medrrg.features.cache_identity import unimed_semantic_identity  # noqa: E402
from medrrg.utils import (now_iso, read_json, sha256_file, sha256_json,  # noqa: E402
                          write_json)

DEFAULT_UNIMED = Path(r"D:\图神经网络\开放词典检测\UniMed-CLIP")
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def _source_tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    for path in sorted((root / "model_configs").glob("*.json")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"期望 [views, concepts]，收到 {tuple(values.shape)}")
    k = max(1, int(np.ceil(values.shape[0] * fraction)))
    return values.topk(k, dim=0).values.mean(dim=0)


def _load_prompts(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload.get("concepts")
    if not concepts:
        raise ValueError("prompts.json 缺少 concepts")
    for concept in concepts:
        if not concept.get("positive") or not concept.get("negative"):
            raise ValueError(f"概念 {concept.get('id')} 缺少正/负提示")
    return payload


def _normalize_slice(slice_xy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(slice_xy, dtype=np.float32)
    mask = slice_xy > 0
    if not mask.any():
        return out
    low, high = np.percentile(slice_xy[mask], [1.0, 99.0])
    out = np.clip((slice_xy - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _to_image(array_xy: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array_xy, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    chest_ct = config.get_path("dataset.modality") == "CT"
    unimed_root = config.path("semantic.unimed_root", str(DEFAULT_UNIMED))
    weights = config.path("semantic.weights", str(unimed_root / "weights" / "unimed-clip-vit-b16.pt"))
    text_encoder = config.path("semantic.text_encoder", str(unimed_root / "weights" / "biomedbert"))
    prompt_path = (Path(args.prompts) if args.prompts else
                   config.path("semantic.prompts",
                               "configs/prompts_brain_mri.json"))
    for path in (weights, text_encoder, prompt_path):
        if not path.exists():
            raise FileNotFoundError(f"UniMed 资源不存在：{path}")
    src = unimed_root / "UniMed-CLIP" / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"UniMed 源码不存在：{src}")

    sys.path.insert(0, str(src))
    from open_clip import HFTokenizer, create_model_and_transforms, get_mean_std  # noqa: E402
    import open_clip.factory as open_clip_factory  # noqa: E402

    def _trusted_local_state_dict(checkpoint_path: str, map_location: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location,
                                weights_only=False)
        state_dict = (checkpoint["state_dict"]
                      if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                      else checkpoint)
        if next(iter(state_dict)).startswith("module"):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        return state_dict

    open_clip_factory.load_state_dict = _trusted_local_state_dict
    load_report: dict[str, list[str]] = {}
    original_load_checkpoint = open_clip_factory.load_checkpoint

    def _tracked_load_checkpoint(model_to_load, checkpoint_path, strict=False):
        incompatible = original_load_checkpoint(model_to_load, checkpoint_path,
                                                strict=strict)
        load_report["missing"] = list(incompatible.missing_keys)
        load_report["unexpected"] = list(incompatible.unexpected_keys)
        return incompatible

    open_clip_factory.load_checkpoint = _tracked_load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("真实 UniMed 语义导出需要 CUDA（CPU 不适合完整缓存）")
    mean, std = get_mean_std()
    model, _, transform = create_model_and_transforms(
        "ViT-B-16-quickgelu", str(weights), precision="amp", device=device,
        force_quick_gelu=True, mean=mean, std=std, inmem=True,
        text_encoder_name=str(text_encoder))
    allowed_unexpected = {"text_encoder.transformer.embeddings.position_ids"}
    bad_unexpected = [key for key in load_report.get("unexpected", [])
                      if key not in allowed_unexpected]
    if load_report.get("missing") or bad_unexpected:
        raise RuntimeError(
            "UniMed checkpoint 与 ViT-B-16-quickgelu 结构不完全匹配："
            f"missing={load_report.get('missing', [])[:8]}, "
            f"unexpected={bad_unexpected[:8]}")
    model.eval()
    tokenizer = HFTokenizer(str(text_encoder), context_length=256)

    prompts = _load_prompts(prompt_path)
    concepts = prompts["concepts"]
    num_concepts = int(config.get_path("semantic.num_concepts", 13))
    if len(concepts) != num_concepts:
        raise ValueError(f"提示词概念数 {len(concepts)} != semantic.num_concepts {num_concepts}")
    with torch.inference_mode():
        positive, negative = [], []
        for concept in concepts:
            pos = model.encode_text(torch.cat([tokenizer(text) for text in
                                               concept["positive"]]).to(device))
            neg = model.encode_text(torch.cat([tokenizer(text) for text in
                                               concept["negative"]]).to(device))
            pos = pos / pos.norm(dim=-1, keepdim=True)
            neg = neg / neg.norm(dim=-1, keepdim=True)
            positive.append(torch.nn.functional.normalize(
                pos.mean(dim=0, keepdim=True), dim=-1))
            negative.append(torch.nn.functional.normalize(
                neg.mean(dim=0, keepdim=True), dim=-1))
        positive = torch.cat(positive).squeeze(1)
        negative = torch.cat(negative).squeeze(1)
        logit_scale = float(model.logit_scale.exp().detach().cpu())

    manifest = load_manifest(config.path("manifest"))
    from medrrg.data.adapters import build_adapter

    adapter = build_adapter(config)
    studies = manifest.studies[: int(args.limit)] if args.limit else manifest.studies
    localization = AdmLocalizationCache.load(config.path("evidence.localization_dir"))
    localization.verify([study.study_id for study in studies],
                        require_predicted=not bool(config.get_path("adm.oracle_mask",
                                                                  False)))

    output = Path(args.output) if args.output else config.path("evidence.semantic_dir")
    (output / "semantic").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_path, output / "prompts.json")
    prompt_sha = sha256_file(output / "prompts.json")
    vendored_sha = _source_tree_sha(src / "open_clip")
    code_sha = sha256_json({"bridge": sha256_file(Path(__file__)),
                            "vendored_open_clip": vendored_sha})
    weight_sha = sha256_file(weights)
    text_sha = (sha256_file(text_encoder / "pytorch_model.bin")
                if (text_encoder / "pytorch_model.bin").is_file()
                else sha256_file(text_encoder / "config.json"))
    modalities = [str(item) for item in config.get_path("visual.modalities",
                                                        DEFAULT_MODALITIES)]
    topk_slices = int(config.get_path("localization.topk_slices", 3))
    top_fraction = float((prompts.get("aggregation") or {}).get("top_fraction", 0.1))
    # 直接用 CLIP 的 logit_scale（~100）乘 margin 会让 probability 全部饱和到 1，
    # 使 F1/ECE 失去意义。首版用预注册的固定温度 20，仅做尺度压缩，不是校准。
    probability_temperature = float(config.get_path(
        "semantic.probability_temperature", 20.0))
    if probability_temperature <= 0:
        raise ValueError("semantic.probability_temperature 必须 > 0")
    sampling = {"topk_slices": topk_slices,
                "source_sequence": str(config.get_path("localization.source_sequence",
                                                      "t2f")),
                "views_per_study": topk_slices * len(modalities),
                "top_fraction": top_fraction,
                "probability_temperature": probability_temperature,
                "use_original_image": True,
                "normalization": "nonzero_p1_p99"}
    if chest_ct:
        sampling.update(normalization="ct_hu_window", orientation=ct_windows.CT_ORIENTATION,
                        windows_hu={g: list(ct_windows.CT_WINDOWS[g]) for g in modalities},
                        ct_window_sha256=sha256_file(Path(ct_windows.__file__)))

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
        invariants = {"format": SEMANTIC_FORMAT,
                      "unimed_code_sha256": code_sha,
                      "unimed_checkpoint_sha256": weight_sha,
                      "text_encoder_sha256": text_sha,
                      "prompt_bank_sha256": prompt_sha,
                      "localization_identity": localization.identity,
                      "sampling": sampling, "concept_ids": [c["id"] for c in concepts]}
        mismatched = [key for key, value in invariants.items()
                      if previous.get(key) != value]
        if mismatched:
            raise RuntimeError(
                f"已有语义缓存与当前设置不一致：{mismatched}；"
                "请更换输出目录或显式使用 --force 重建。")

    image_hashes: dict[str, dict[str, str]] = {}
    for index, study in enumerate(studies):
        semantic_path = output / "semantic" / f"{study.study_id}.npz"
        location = localization.get(study.study_id)
        shape_z = int(location.shape_zyx[0])
        selected_z = [int(z) for z in location.selected_z[:topk_slices]]
        current_hashes: dict[str, str] = {}
        paths: dict[str, Path] = {}
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
        if semantic_path.exists() and study.study_id in existing and not args.force:
            previous = dict((existing[study.study_id].get("input_sha256") or {}))
            if previous != current_hashes:
                raise RuntimeError(
                    f"{study.study_id}: 输入影像已变化；请使用 --force 重建语义缓存。")
            missing = missing_npz_keys(semantic_path, SEMANTIC_ARRAYS)
            if missing:
                raise RuntimeError(
                    f"{study.study_id}: 语义缓存缺少字段 {missing}；"
                    "请使用 --force 重建语义缓存。")
            print(f"[skip] {study.study_id}", flush=True)
            rows.append(existing[study.study_id])
            continue

        import nibabel as nib

        volumes = {path: np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj,
                                       dtype=np.float32) for path in set(paths.values())}
        tensors = []
        for z in selected_z:
            for group in modalities:
                volume = volumes[paths[group]]
                if not 0 <= z < volume.shape[2]:
                    raise ValueError(f"{study.study_id}: z={z} 超出 {volume.shape[2]}")
                axial = volume[:, :, z]
                original = ct_windows.ct_window(axial, group) if chest_ct else _normalize_slice(axial)
                tensors.append(transform(_to_image(original)))
        batch = torch.stack(tensors).to(device)
        concept_ids = [concept["id"] for concept in concepts]
        id_to_index = {concept_id: i for i, concept_id in enumerate(concept_ids)}
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            positive_scores = features @ positive.T
            negative_scores = features @ negative.T
            pos = _top_fraction_mean(positive_scores, top_fraction)
            margins = _top_fraction_mean(positive_scores - negative_scores, top_fraction)
            probabilities = torch.sigmoid(probability_temperature * margins)
            confidence = torch.abs(2 * probabilities - 1)
            # 互斥组（肿瘤类型）在 inference_mode 内就地改写，避免 inference tensor
            # 在 inference_mode 之外被 inplace 更新。
            for group in prompts.get("exclusive_groups", []):
                indices = [id_to_index[item] for item in group.get("concept_ids") or []
                           if item in id_to_index]
                if len(indices) < 2:
                    continue
                group_similarity = _top_fraction_mean(features @ positive[indices].T,
                                                      top_fraction)
                group_probabilities = torch.softmax(
                    probability_temperature * group_similarity, dim=0)
                gate_id = group.get("gate_concept")
                if gate_id:
                    if gate_id not in id_to_index:
                        raise ValueError(f"互斥组 {group.get('id')} 的 gate_concept 不存在")
                    group_probabilities = (group_probabilities
                                           * probabilities[id_to_index[gate_id]].detach())
                for local_index, concept_index in enumerate(indices):
                    competitors = torch.cat([group_similarity[:local_index],
                                             group_similarity[local_index + 1:]])
                    probabilities[concept_index] = group_probabilities[local_index]
                    margins[concept_index] = (group_similarity[local_index]
                                              - competitors.max())
                    confidence[concept_index] = group_probabilities[local_index]

        np.savez_compressed(
            semantic_path,
            concept_ids=np.asarray(concept_ids),
            text_prototypes=positive.cpu().numpy().astype(np.float32),
            probabilities=probabilities.cpu().numpy().astype(np.float32),
            confidence=confidence.cpu().numpy().astype(np.float32),
            margins=margins.cpu().numpy().astype(np.float32))
        rows.append({
            "study_id": study.study_id,
            "array_path": f"semantic/{study.study_id}.npz",
            "selected_z": [int(z) for z in selected_z],
            "num_views": len(tensors),
            "input_sha256": current_hashes,
        })
        print(f"[unimed] {index + 1}/{len(studies)} {study.study_id} "
              f"views={len(tensors)}", flush=True)

    with open(output / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    identity = unimed_semantic_identity(
        code_sha, weight_sha, text_sha, prompt_sha, localization.identity,
        sampling, image_hashes)
    meta = {
        "format": SEMANTIC_FORMAT,
        "identity": identity,
        "localization_identity": localization.identity,
        "unimed_code_sha256": code_sha,
        "vendored_open_clip_sha256": vendored_sha,
        "unimed_checkpoint_sha256": weight_sha,
        "text_encoder_sha256": text_sha,
        "prompt_bank_sha256": prompt_sha,
        "classification_mode": "zero_shot",
        "d_proto": 512,
        "concept_ids": [c["id"] for c in concepts],
        "sampling": sampling,
        "studies": sorted(row["study_id"] for row in rows),
        "image_sha256": image_hashes,
        "created_at": now_iso(),
        "notes": "语义词向量为真实 UniMed 图文相似度；probability 是未校准相似度，"
                 "不是临床诊断概率。",
    }
    write_json(output / "meta.json", meta)
    print(f"[unimed] 写入 {len(rows)} 例 -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
