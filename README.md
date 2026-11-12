# MedADM-RRG：少样本异常检测 + UniMed-CLIP 语义 + ADM 增强 + 报告 Transformer

对应方案：`docs/EKAGen式异常增强_ResNet_UniMed_Transformer简化开发方案.md`
（v3.0；与仓库上一级的同名文件内容一致）。本目录是该方案的可运行工程实现。

> 正式主线只有一条：
> **冻结的少样本异常检测器产生注意力图 → ADM 增强原图 → 冻结 ResNet-101 提取
> 视觉 patch tokens → 冻结 UniMed-CLIP 给出疾病语义 tokens → 两类 token 拼接后
> 由小型 Encoder–Decoder 生成 Findings + Impression。**
>
> 没有独立 ROI token 支路、没有预测框位置/几何 token、没有 QueryResampler、
> 不做三个大模型端到端联合训练。所有上游证据都是**预测**，不用于临床诊断。

---

## 1. 数据流

```text
四序列脑 MRI（t1n/t1c/t2w/t2f，NIfTI）
  │
  ├─ 冻结少样本异常检测器（最新定位头版 mri_foreground_k5）
  │     → 预测热图 heatmap_3d[Z,heat,heat] + score_per_slice[Z] + abnormal_score
  │
  ├─ ADM 增强
  │     A = normalize(resize(heatmap))
  │     Iadm = I × [γ + (1−γ)·A]          默认 γ=0.4, threshold=0.6, soft
  │     abnormal_score < threshold → Iadm = I（正常回退）
  │     Oracle（O0）：GT mask 直接二值增强，不做置信度回退
  │
  ├─ 冻结 ResNet-101（ImageNet，删 avgpool/fc）
  │     C5[2048,7,7] → AdaptiveAvgPool2d(3,3) → [9,2048]
  │     → Linear(2048,256)+LayerNorm → 9 patch tokens
  │     3 个异常切片 × 4 序列 × 9 patch = 108 visual tokens
  │
  └─ 冻结 UniMed-CLIP（ViT-B-16 + BiomedBERT，零样本）
        13 个概念的正/负提示原型 → 原图相似度 → probability/confidence/margin
        → 13 个 semantic tokens（prototype 512→256 + 统计 MLP + confidence gate）
  │
  [108 visual; 13 semantic] = 121 evidence tokens
  → modality / normalized-z / view / type / **2D patch(row,col)** embedding
  → 2 层 Encoder（d=256, 8 heads, ffn=1024）
  → 4 层 Decoder（max_report_tokens=256）
  → <FINDINGS>…</FINDINGS> <IMPRESSION>…</IMPRESSION> <EOS>
```

只有两类证据 token：`visual_tokens` 与 `semantic_tokens`。

关于实现细节（可复现性/可评价性所必需）：

* **视觉投影确定性**：缓存保存的是 `Linear(2048→256)+LayerNorm` 投影后的 tokens，
  该投影由 `visual.proj_seed`（默认 0）确定性初始化，其 SHA256 写入缓存 meta 与
  身份链。否则不同导出进程的 tokens 不可比，`null_adm` 与 B0/B1 缓存也无法对齐。
* **2D patch 位置编码**：同一张切片的 9 个 patch 只有 modality/z/view 时没有空间
  区分，打乱后模型看到同一集合；因此缓存显式保存 `patch_row/patch_col`，模型为
  行/列各加一个 embedding（ViT 式 2D 位置编码）。
* **语义 probability 温度**：UniMed 的 `logit_scale` 约 100，直接
  `sigmoid(logit_scale × margin)` 会让 13 个概念的概率几乎全部饱和到 1，
  使 F1/ECE 失去意义。首版使用预注册的固定温度
  `semantic.probability_temperature=20`，只做尺度压缩，**不是校准**；
  margin 与文本原型同样写入缓存，AUROC 等排序指标不受影响。

---

## 2. 目录

```text
MedADM_RRG/
  medrrg/
    config.py, contracts.py, runtime_guard.py, ontology.py, evidence.py
    data/
      manifest.py, adapters/{base,radgenome_brain_mri}.py
      adm_dataset.py            # 词表 + ADM 证据数据集 + collate
    features/
      adm_enhancer.py           # 热图归一化 / soft·hard ADM / 正常回退 / top-k 切片
      resnet_visual.py          # ResNet-101 patch 编码器（C5→3×3→2048→256）
      adm_cache.py              # 三类缓存读取与严格校验
      cache_identity.py         # 定位 → 视觉 → 语义 → checkpoint 身份链
    model/
      token_projectors.py       # SemanticProjector + modality/z/view/type embedding
      transformer_encoder.py, transformer_decoder.py
      report_model.py           # AdmEvidenceReportTransformer（evidence_mode=adm_simple）
    training/{checkpoint,trainer}.py
    generation/generate.py
    evaluation/{nlg_metrics,concept_metrics,evidence_intervention,upstream,run}.py
  external_tools/
    export_adm_localization.py  # step02a：真实定位头 → 预测热图缓存
    export_resnet_visual.py     # step02b：ADM + ResNet → 108×256 tokens
    export_unimed_semantic.py   # step02c：UniMed → 13 概念语义
  scripts/step01..step05, step02a..step02d, make_pilot_subset.py
  configs/
    adm_data.yaml, adm_model.yaml
    adm_radgenome_b0.yaml, adm_radgenome_b1.yaml,
    adm_radgenome_f0.yaml, adm_radgenome_oracle.yaml
    prompts_brain_mri.json
    pilot/pilot_f0.yaml          # 4 例真实病例的全链路自检配置
  tests/                        # 方案 §10 要求的用例
  docs/                         # 实施说明与本地验证记录
  caches/ checkpoints/ results/ # 运行产物（不入库）
```

---

## 3. 环境

已在本机 `torch-gpu1` 环境验证：

```powershell
$py = 'C:\Users\HP\.conda\envs\torch-gpu1\python.exe'
Set-Location 'D:\图神经网络\医学文本报告生成\融合尝试Transformer版\MedADM_RRG'
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONIOENCODING = 'utf-8'
& $py -m pip install -e . --no-deps
& $py -m medrrg.cli doctor --config configs\adm_radgenome_f0.yaml
```

依赖：torch、torchvision、numpy、scipy、nibabel、PyYAML、Pillow、transformers。
ResNet-101 权重使用本机 torchvision 缓存（`resnet101-63fe2227.pth`）；UniMed-CLIP
需要 `unimed-clip-vit-b16.pt`（约 2.35 GB）与 `weights/biomedbert`。

---

## 4. 上游资源

| 资源 | 位置 |
|---|---|
| 定位源码（少样本异常检测） | `D:\图神经网络\小样本原型学习\最新定位头版` |
| MRI 定位 checkpoint | `runs/mri_foreground_k5_20261029/checkpoint_selected.pt` |
| UniMed 源码 | `D:\图神经网络\开放词典检测\UniMed-CLIP\UniMed-CLIP` |
| UniMed 权重 | `weights/unimed-clip-vit-b16.pt` |
| 文本编码器 | `weights/biomedbert` |
| 数据 | `D:\图神经网络\小样本原型学习\数据集\RadGenome-Brain_MRI\BraTS_GLI` |

定位头只在 K5 少样本设置下训练，MRI 小病灶召回偏低；UniMed 会把部分胶质瘤
判成“无肿瘤”。两者都是**不确定证据**，不是 GT 或硬标签。

---

## 5. 缓存合同

```text
caches/adm/localization/<run>/
  meta.json  index.jsonl  arrays/<study_id>.npz
    heatmap_3d[Z,heat,heat] float16 / score_per_slice[Z] float32 / abnormal_score
caches/adm/resnet/<run>/                      # 格式 medadm.adm.resnet.v2
  meta.json  index.jsonl  visual/<study_id>.npz
    visual_tokens[108,256] / visual_tokens_null[108,256] / visual_mask[108]
    visual_modality[108] / visual_z[108] / visual_view[108]
    visual_patch_row[108] / visual_patch_col[108]   # 3×3 网格行/列（2D 位置）
    selected_z[3]
caches/adm/unimed/<run>/
  meta.json  index.jsonl  prompts.json  semantic/<study_id>.npz
    concept_ids[13] / text_prototypes[13,512] / probabilities[13]
    confidence[13] / margins[13]
```

身份链：

```text
定位（权重+预处理+热图网格+阈值+影像哈希）
  → 视觉（定位 identity + ResNet 权重 + γ/threshold/mode + 切片策略 + patch 网格）
  → 语义（UniMed 权重/文本塔/提示词 + 定位 identity + 采样策略 + 影像哈希）
  → 报告 checkpoint（+ tokenizer + 模型配置 + 训练配置 + 源码树哈希）
```

任何一层变化都会使训练/生成/评价直接失败，不会静默复用旧缓存。
`visual_tokens_null` 是同一 ResNet 对**原图**的编码，用于 `null-adm` 干预
（等价于 Iadm=I），不需要重新训练或重跑 ResNet。

> 视觉缓存自 `v2` 起必须包含 `visual_patch_row/col`：报告模型用它们加 2D patch
> 位置编码。旧缓存（v1）会被格式校验拒绝，需要重新运行 `step02b`。

---

## 6. 正式命令（方案 §12）

```powershell
$py = 'C:\Users\HP\.conda\envs\torch-gpu1\python.exe'
$proj = 'D:\图神经网络\医学文本报告生成\融合尝试Transformer版\MedADM_RRG'
Set-Location $proj
$env:PYTHONPATH = $proj
$env:PYTHONIOENCODING = 'utf-8'

# 1) manifest（230 例，161/23/46）
& $py scripts\step01_build_manifest.py --config configs\adm_radgenome_f0.yaml

# 2) 定位热图（B0/B1/F0 共用；可加 --limit N 只跑 pilot）
& $py scripts\step02a_export_adm.py --config configs\adm_radgenome_f0.yaml

# 3) 视觉 tokens：B0 用原图，B1/F0 用 ADM 增强图（两个目录，互不覆盖）
& $py scripts\step02b_export_resnet.py --config configs\adm_radgenome_b0.yaml
& $py scripts\step02b_export_resnet.py --config configs\adm_radgenome_f0.yaml

# 4) UniMed 语义（13 概念，使用原图）
& $py scripts\step02c_export_unimed_semantic.py --config configs\adm_radgenome_f0.yaml

# 5) 校验身份链 / 上游指标
& $py scripts\step02d_validate_evidence.py --config configs\adm_radgenome_f0.yaml --evaluate

# 6) 训练 B0/B1/F0（8–16 例过拟合自检用 --overfit-cases 8；
#    --epochs / --lr / --batch-size / --out 可临时覆盖配置）
& $py scripts\step03_train.py --config configs\adm_radgenome_b0.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_b1.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_f0.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_f0.yaml --overfit-cases 8

# 7) 生成 + 评价
& $py scripts\step04_generate.py --checkpoint checkpoints\adm_f0_resnet_unimed\best.pt --split test --max-len 256 --out results\adm_f0\test
& $py scripts\step05_evaluate.py --run results\adm_f0\test --interventions --concept-deltas
```

Oracle 上限（O0，只用于上限分析，`runtime.formal: false`，独立目录）：

```powershell
& $py scripts\step02a_export_adm.py --config configs\adm_radgenome_oracle.yaml
& $py scripts\step02b_export_resnet.py --config configs\adm_radgenome_oracle.yaml
& $py scripts\step02c_export_unimed_semantic.py --config configs\adm_radgenome_oracle.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_oracle.yaml
```

---

## 7. 实验矩阵

| 实验 | 视觉输入 | UniMed 语义 | 配置 | 目的 |
|---|---|---|---|---|
| B0 | 原图 → ResNet | 否 | `adm_radgenome_b0.yaml` | 基础报告模型 |
| B1 | ADM 增强图 → ResNet | 否 | `adm_radgenome_b1.yaml` | 异常增强贡献 |
| F0 | ADM 增强图 → ResNet | 是 | `adm_radgenome_f0.yaml` | 完整模型 |
| O0 | GT mask 增强图 → ResNet | 是 | `adm_radgenome_oracle.yaml` | 定位误差上限 |

```text
B1 − B0 = 异常检测 + ADM 的贡献
F0 − B1 = UniMed 疾病语义的贡献
O0 − F0 = 定位误差造成的上限差距
```

没有 ROI/几何位置 token，所以不需要位置 token 消融；patch 内部的位置信息由
`AdmEvidenceAssembler` 的 2D 行列 embedding 提供（v2 缓存显式保存 patch 行列）。

---

## 8. 评价

* 上游定位/ADM：Dice、lesion recall、attention 覆盖率、病灶/背景响应比、
  像素 AUROC、注意力峰是否落在病灶内（`step02d --evaluate`）；
* UniMed：每概念 AUROC/AUPRC/F1/ECE、macro AUROC/F1、互斥肿瘤类型准确率；
* 报告：BLEU-1/2/3/4、ROUGE-L、CIDEr-D、METEOR-lite、概念 P/R/F1、
  概念级幻觉率、Findings/Impression 分段完整率；
* 证据干预：`null_adm`、`swap_adm`、`null_semantic`、`swap_semantic`
  （患者级 bootstrap：干预后 NLL 应上升）。

---

## 9. 测试

```powershell
& $py -m pytest -q
```

覆盖方案 §10：

| 用例 | 内容 |
|---|---|
| `test_adm_enhancer.py` | 热图范围、γ 边界、正常回退、背景不完全清零、hard 模式、**Oracle 小病灶不回退** |
| `test_resnet_visual.py` | C5 形状、3×3 patch 池化、2048→256 投影、冻结、投影确定性 |
| `test_adm_dataset.py` | 只有 visual/semantic、无 ROI/geometry、padding 与 mask、teacher forcing、**patch 行列** |
| `test_adm_report_model.py` | memory=visual+semantic、B0/B1 无语义、13 语义 token、EOS、**2D 位置编码生效** |
| `test_adm_cache_identity.py` | 权重/γ/threshold/prompt 变化使缓存失效 |
| `test_ontology.py` | **否定作用域：句内、后置否定不跨分句、并列主语、对比连词、中文** |
| `test_nlg_cider.py` | **CIDEr-D 官方公式回归（满分 10、零重叠 0、长度惩罚）** |
| `test_export_guards.py` | **v1 meta / 缺 patch 字段的旧 NPZ 在导出复用前被拒绝** |
| `test_pipeline_smoke.py` | 合成缓存上训练→生成→评价→四种干预全链路 |
| `test_manifest.py` | manifest 哈希与患者独立 split |

真实数据上的验证记录（230 例缓存、上游指标、短训练与干预）见
[`docs/本地验证记录.md`](docs/本地验证记录.md)。当前状态：

* 230 例定位热图与 13 概念语义缓存已真实导出；视觉缓存升级到 v2 后，旧的 v1
  缓存已删除，需要重新运行 `step02b`（两次，B0/B1）；
* 证据身份链校验通过，**74 个测试通过**；
* B0/B1/F0 已完成冒烟级短训练（6–8 epoch），生成/评价/干预链路跑通；
* 8 例过拟合自检把 val_loss 从 4.65 降到 **0.119**，训练病例上 BLEU-4 0.881、
  概念 F1 1.0、分段完整率 8/8，说明模型结构与训练链路正确；
* P1/P2 修复（Oracle 回退、patch 位置编码、CIDEr-D、否定作用域、后置否定分句、
  导出格式守卫）改变了视觉缓存格式与指标口径：v1 视觉缓存与旧 checkpoint 已失效，
  需按第 6 节重新导出/训练；
* **尚未跑论文级 100 epoch × 3 种子与 O0 上限**，现有测试集数字不能当作结果。

---

## 10. 已知限制（必须如实写入论文/答辩）

1. **报告模型只读缓存**：定位器、ResNet、UniMed 都不在训练图中，联合微调不在首版范围。
2. **定位误差是上限瓶颈**：MRI 小病灶召回偏低，O0 用来量化该差距。
3. **UniMed 相似度未校准**：`probabilities` 不是临床诊断概率；拒识阈值需独立验证集。
4. **视觉 token 只来自 3 个切片**：多切片/三维形态的疾病区分能力有限。
5. **CIDEr-D 已按官方 `cider_scorer` 公式实现**；METEOR 仍是 meteor-lite，
   正式论文数字应换官方 METEOR 并记录包版本。
6. **否定识别是规则启发式**：已限定句内、后置否定紧贴自身谓词且不跨分句、支持并列
   主语共享谓语，但仍无法覆盖全部双重否定/长距离依存；概念指标应同时报告原始
   阳性提及数。
7. **患者独立性**：RadGenome-Brain MRI 的 split 已按患者划分并通过 manifest 校验，
   但原始数据划分的临床合理性仍需论文说明。

---

## 11. 与旧版 `融合尝试/MedEvidence_RRG` 的区别

| 维度 | 旧版 | 本版 |
|---|---|---|
| 视觉输入 | 全图 + 预测 ROI 两条支路 | 只有一条：ADM 增强图 |
| 视觉编码器 | UniMed-CLIP 图像塔 | ResNet-101（ImageNet） |
| token 数 | 32 全图 + 12 ROI + 3 几何 + 4 语义 | 108 视觉 + 13 语义 |
| 位置/几何 | box 坐标 + GeometryProjector | 无 |
| 重采样 | global/ROI QueryResampler | 无（后续才做扩展消融） |
| 语义 | 4 个 Top-K 概念 | 13 个概念全部保留 |
| 阶段 | 全部一次性训练 | 先缓存，再训练报告模型 |


## 公开源码目录

本目录收录项目源码、配置、测试、运行文档和定位器依赖源码。运行前按上文及对应流程文档准备原始数据、模型权重和缓存，并将配置里的路径换成自己的位置。MRI 定位器的源码也可在本仓库的 vendor/paclip 下找到。K=32 历史实验入口为 external_tools/fit_mri_support_2d.py，其对应配置在 configs/adapt，路径需按实际训练环境填写。可用 scripts/download_radgenome_glioma.py 准备 MRI 数据。胸部 CT 流程见 [ChestCT 训练方案](docs/ChestCT_K5_训练方案.md)。
