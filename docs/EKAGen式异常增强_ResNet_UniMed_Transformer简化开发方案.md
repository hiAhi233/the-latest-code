# MedEvidence-RRG 简化开发方案

## EKAGen 式异常增强 + ResNet 视觉特征 + UniMed 语义 + Transformer 报告生成

> 版本：v3.0，2026-11-10  
> 项目：D:\图神经网络\医学文本报告生成\融合尝试\MedEvidence_RRG  
> 本文取代此前“全图 Token + ROI Token + 位置 Token”的正式开发方案。

---

## 1. 只做什么

正式主线只有一条：

~~~text
医学图像
  ↓
异常检测器产生异常注意力图
  ↓
注意力图增强原图
  ↓
ResNet 提取视觉 patch tokens
  ↓
与 UniMed 疾病语义 tokens 拼接
  ↓
Transformer 生成 Findings + Impression
~~~

不再实现：

- 不建立独立 ROI token 支路；
- 不把预测框坐标编码成位置 token；
- 不使用 GeometryProjector；
- 不分别编码全图和 ROI；
- 不把 QueryResampler 作为首版必需组件；
- 不做三个大模型端到端联合训练。

异常定位器只负责生成注意力图。注意力图直接作用于原图，视觉侧最终只有一套 ResNet patch tokens。这对应 EKAGen 的 ADM → Visual Extractor → Flattened Patches 思路。

---

## 2. 最终结构

~~~text
                    ┌──────────────────────────────┐
                    │       原始医学图像 I         │
                    │  T1 / T1C / T2 / FLAIR      │
                    └──────────────┬───────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │ 异常检测器（冻结）  │
                        └──────────┬──────────┘
                                   │
                          异常热图 A / score
                                   │
                        ┌──────────▼──────────┐
                        │ ADM 式注意力增强    │
                        │ I × [γ+(1-γ)A]      │
                        └──────────┬──────────┘
                                   │
                            异常增强图像 Iadm
                                   │
                        ┌──────────▼──────────┐
                        │ ResNet-101          │
                        │ Visual Extractor    │
                        └──────────┬──────────┘
                                   │
                         C5 patch feature map
                                   │
                        ┌──────────▼──────────┐
                        │ Pool + Linear       │
                        │ 2048 → 256          │
                        └──────────┬──────────┘
                                   │
                             Visual Tokens
                                   │
                                   ├──────────────────┐
                                   │                  │
原始图像 → UniMed-CLIP → 疾病概率/文本原型            │
                                   │                  │
                            Semantic Projection       │
                                   │                  │
                             Semantic Tokens          │
                                   │                  │
                                   └────────┬─────────┘
                                            │
                              [Visual Tokens; Semantic Tokens]
                                            │
                                  Transformer Encoder
                                            │
                                  Transformer Decoder
                                            │
                                Findings + Impression
~~~

只有两类证据 token：

1. visual_tokens：异常增强图像经过 ResNet 得到；
2. semantic_tokens：UniMed 疾病判断经过语义投影得到。

---

## 3. 模块职责

### 3.1 异常检测器

输入二维切片，输出：

~~~text
heatmap          [H,W]
abnormal_score   scalar
~~~

首个脑 MRI 通路继续复用：

~~~text
D:\图神经网络\小样本原型学习\最新定位头版
runs\mri_foreground_k5_20261029\checkpoint_selected.pt
~~~

正式路径只使用预测热图。GT segmentation 只用于定位评价和 Oracle 上限，不能进入主结果。

报告训练阶段冻结异常检测器。

### 3.2 ADM 式异常增强

把热图重采样到图像大小并归一化到 [0,1]：

~~~text
A = normalize(heatmap)
Iadm = I × [γ + (1-γ) × A]
~~~

默认配置：

~~~yaml
adm:
  gamma: 0.4
  threshold: 0.6
  mode: soft
  normal_fallback: original
~~~

作用：

- 高异常区域基本保留；
- 正常背景缩放到约 40%；
- 不做硬裁剪，保留脑室、中线、侧别和整体解剖；
- 异常置信度低于正常阈值时直接令 Iadm=I，避免把正常图整体压暗。

四序列 MRI 使用空间对齐后的同一注意力图：

~~~text
T1    × A → T1_adm
T1C   × A → T1C_adm
T2    × A → T2_adm
FLAIR × A → FLAIR_adm
~~~

首版优先使用 FLAIR 定位热图映射到四个已配准序列。

### 3.3 ResNet 视觉特征提取器

首版使用：

~~~text
ImageNet 预训练 ResNet-101
删除 avgpool 和 fc
输出 C5 feature map：[2048,7,7]
~~~

灰度图复制为三通道，沿用预训练 ResNet 的输入归一化。

视觉 token：

~~~text
C5 [2048,7,7]
  ↓ AdaptiveAvgPool2d(3,3)
[2048,3,3]
  ↓ flatten spatial
[9,2048]
  ↓ Linear(2048,256) + LayerNorm
[9,256]
~~~

脑 MRI 每例选择异常分数最高的 3 个 z 切片，每层使用四个序列：

~~~text
3 slices × 4 modalities × 9 patches = 108 visual tokens
~~~

每个 token 加入：

- modality embedding：T1/T1C/T2/FLAIR；
- normalized z embedding；
- view embedding；
- visual type embedding。

首版直接把 108 个 patch tokens 送入 Transformer，不使用 QueryResampler。以后只有在 token 数过大时才把 Resampler 作为扩展消融。

### 3.4 UniMed 疾病语义

UniMed 只负责开放词汇疾病和征象识别：

- 不承担报告视觉特征提取；
- 不产生位置 token；
- 不输出 ROI token。

输入使用与视觉支路相同的代表性原图。首版不用 ADM 图作为 UniMed 输入，避免改变其预训练分布。

输出：

~~~text
concept_id
text_prototype [512]
probability
confidence
margin
~~~

本地权重：

~~~text
D:\图神经网络\开放词典检测\UniMed-CLIP\weights\unimed-clip-vit-b16.pt
D:\图神经网络\开放词典检测\UniMed-CLIP\weights\biomedbert
~~~

语义投影：

~~~text
text_prototype [512] → Linear(512,256)
[probability, confidence, margin] → MLP(3,256,256)
两者相加并经过 confidence gate
→ semantic token [256]
~~~

首版保留全部 13 个概念，因此得到 13×256 semantic tokens。

具体肿瘤类型使用互斥组；水肿、出血、坏死、强化和占位效应使用多标签分数。

### 3.5 报告 Transformer

输入：

~~~text
108 visual tokens
+ 13 semantic tokens
= 121 evidence tokens
~~~

Encoder：

~~~yaml
d_model: 256
heads: 8
layers: 2
ffn: 1024
dropout: 0.1
~~~

Decoder：

~~~yaml
d_model: 256
heads: 8
layers: 4
ffn: 1024
max_report_tokens: 256
~~~

训练与生成目标：

~~~text
<FINDINGS> ... </FINDINGS>
<IMPRESSION> ... </IMPRESSION>
<EOS>
~~~

---

## 4. 缓存合同

### 4.1 定位缓存

~~~text
caches/adm/localization/<identity>/
  meta.json
  index.jsonl
  arrays/<study_id>.npz
~~~

NPZ：

~~~text
heatmap_3d       [Z,H,W] float16
score_per_slice  [Z] float32
abnormal_score   scalar
~~~

预测框可保留用于定位评价和可视化，但不进入报告模型。

### 4.2 ADM 元数据

不永久保存全部增强 PNG，只保存生成参数和切片索引：

~~~json
{
  "study_id": "...",
  "selected_z": [54, 55, 56],
  "source_heatmap_identity": "...",
  "gamma": 0.4,
  "threshold": 0.6,
  "mode": "soft"
}
~~~

ResNet 特征导出时生成 Iadm，随后只缓存视觉特征。

### 4.3 视觉缓存

~~~text
caches/adm/resnet/<identity>/
  meta.json
  index.jsonl
  visual/<study_id>.npz
~~~

每例：

~~~text
visual_tokens    [Nv,256]
visual_mask      [Nv]
modality_ids     [Nv]
z_positions      [Nv]
~~~

脑 MRI 默认 Nv=108。

### 4.4 语义缓存

~~~text
caches/adm/unimed/<identity>/
  meta.json
  prompts.json
  semantic/<study_id>.npz
~~~

每例：

~~~text
concept_ids       [13]
text_prototypes   [13,512]
probabilities     [13]
confidence        [13]
margins           [13]
~~~

缓存 identity 必须覆盖定位权重、ResNet 权重、UniMed 权重、prompt bank、gamma、threshold、切片策略和原始影像哈希。

---

## 5. 代码改造

### 5.1 新增文件

~~~text
medrrg/features/
  adm_enhancer.py
  resnet_visual.py
  adm_cache.py

medrrg/data/
  adm_dataset.py

external_tools/
  export_adm_localization.py
  export_resnet_visual.py
  export_unimed_semantic.py

scripts/
  step02a_export_adm.py
  step02b_export_resnet.py
  step02c_export_unimed_semantic.py
  step02d_validate_evidence.py
~~~

### 5.2 adm_enhancer.py

职责：

- 热图归一化；
- 热图与原图空间对齐；
- soft/hard ADM；
- 正常病例回退原图；
- 选择 top-k z；
- 输出原图、热图和增强图可视化。

目标接口：

~~~python
enhanced, meta = enhance_image(
    image,
    heatmap,
    gamma=0.4,
    threshold=0.6,
    abnormal_score=None,
)
~~~

### 5.3 resnet_visual.py

模型：

~~~python
class ResNetPatchEncoder(nn.Module):
    # ResNet-101 without avgpool/fc
    # AdaptiveAvgPool2d((3,3))
    # Linear(2048,256)
    # LayerNorm(256)
~~~

职责：

- 加载预训练 ResNet-101；
- 删除分类头；
- 输出 C5；
- 3×3 patch 池化；
- 2048→256 投影；
- 写 visual cache。

### 5.4 adm_dataset.py

正式 batch 只包含：

~~~text
visual_tokens
visual_mask
visual_modality_ids
visual_z_positions
semantic_prototypes
semantic_scores
semantic_confidence
semantic_margins
semantic_mask
target_ids
~~~

正式路径删除：

~~~text
roi_features
roi_mask
roi_coords
geometry
geometry_mask
geometry_coords
roi_available
~~~

### 5.5 报告模型

新增 AdmEvidenceReportTransformer，或在现有模型增加 evidence_mode=adm_simple：

~~~text
visual tokens → 加 modality/z/type embedding
semantic evidence → SemanticProjector
[visual; semantic] → EvidenceEncoder → ReportDecoder
~~~

ADM 正式路径不实例化：

- global_resampler；
- roi_resampler；
- GeometryProjector；
- ROI null token；
- geometry token type。

旧模块可为历史测试保留，但不能进入新正式配置。

---

## 6. 配置

新增：

~~~text
configs/adm_radgenome_b0.yaml
configs/adm_radgenome_b1.yaml
configs/adm_radgenome_f0.yaml
configs/adm_radgenome_oracle.yaml
~~~

F0 示例：

~~~yaml
experiment:
  name: adm_f0_resnet_unimed

runtime:
  formal: true
  evidence_mode: adm_simple

localization:
  backend: external
  checkpoint: D:/图神经网络/小样本原型学习/最新定位头版/runs/mri_foreground_k5_20261029/checkpoint_selected.pt
  source_sequence: t2f
  topk_slices: 3

adm:
  mode: soft
  gamma: 0.4
  threshold: 0.6
  normal_fallback: original

visual:
  backbone: resnet101
  pretrained: imagenet
  frozen: true
  patch_grid: 3
  output_dim: 256
  modalities: [t1n, t1c, t2w, t2f]

semantic:
  backend: unimed
  frozen: true
  num_concepts: 13
  use_original_image: true

model:
  d_model: 256
  heads: 8
  encoder_layers: 2
  decoder_layers: 4
  ffn: 1024
  max_report_tokens: 256
  use_semantic: true
  use_geometry: false
  use_roi_branch: false

training:
  batch_size: 8
  grad_accum: 2
  epochs: 100
  lr: 2.0e-4
  weight_decay: 0.01
  early_stopping_patience: 12
  amp: true
  seed: 0
~~~

---

## 7. 训练步骤

### A. 异常热图

- 冻结异常检测器；
- 为全部病例导出预测热图；
- 在 train/val segmentation 上确定 ADM threshold 和 gamma；
- test 不参与参数选择。

### B. 视觉缓存

- 冻结 ResNet-101；
- 根据预测热图生成异常增强图；
- 取 top-3 异常切片和四序列；
- 导出 108×256 visual tokens；
- 报告训练期间不再运行 ResNet。

### C. UniMed 语义缓存

- 冻结 UniMed；
- 使用代表性原图做零样本分类；
- 导出 13 个语义概念；
- train/val 可用于提示词确认和温度校准；
- test 只评价。

### D. 报告生成器

冻结：

~~~text
异常检测器
ResNet-101
UniMed-CLIP
~~~

训练：

~~~text
visual patch projection
modality/z/type embeddings
SemanticProjector
2层 Transformer Encoder
4层 Transformer Decoder
~~~

首版只使用报告 token 交叉熵，不加入几何损失、ROI 对比损失或强化学习。

只有 F0 有效后，才考虑：

- 解冻 ResNet layer4；
- 更大的 patch grid；
- EKAGen 式专家报告知识检索；
- teacher self-distillation。

---

## 8. 核心实验

| 实验 | 视觉输入 | UniMed语义 | 目的 |
|---|---|---|---|
| B0 | 原图 → ResNet | 否 | 基础报告模型 |
| B1 | ADM增强图 → ResNet | 否 | 异常增强贡献 |
| F0 | ADM增强图 → ResNet | 是 | 完整模型 |
| O0 | GT mask增强图 → ResNet | 是 | Oracle上限 |

解释：

~~~text
B1 - B0 = 异常检测/ADM 的贡献
F0 - B1 = UniMed 疾病语义的贡献
O0 - F0 = 定位误差造成的上限差距
~~~

没有位置 token，因此不需要位置 token 消融。

---

## 9. 评价

异常检测和 ADM：

- Dice；
- lesion recall；
- attention 覆盖率；
- 病灶/背景响应比；
- 20 例原图、热图、增强图可视化。

UniMed：

- AUROC；
- AUPRC；
- macro/micro F1；
- ECE；
- 互斥肿瘤类型准确率。

报告：

- BLEU-1/2/3/4；
- ROUGE-L；
- METEOR；
- CIDEr-D；
- 临床概念 Precision/Recall/F1；
- hallucination rate；
- Findings/Impression 分段完整率。

证据干预只保留：

- null-adm：令 Iadm=I；
- swap-adm：交换病例注意力图；
- null-semantic：屏蔽 UniMed tokens；
- swap-semantic：交换 UniMed tokens。

---

## 10. 测试

新增：

~~~text
test_adm_enhancer.py
  热图范围、gamma边界、正常回退、背景不完全清零

test_resnet_visual.py
  C5形状、3×3 patch池化、2048→256投影、四模态/z顺序

test_adm_dataset.py
  只有visual和semantic两类token
  不包含ROI/geometry字段
  padding与mask正确

test_adm_report_model.py
  memory长度=visual+semantic
  B0/B1不含semantic
  F0包含13个semantic token
  teacher forcing与EOS正确

test_adm_cache_identity.py
  定位权重、gamma、threshold、ResNet权重和prompt变化均使缓存失效
~~~

formal 配置必须拒绝：

~~~text
geometry tokens
ROI token branch
gt_oracle（主结果）
hash/probe/tiny占位后端
~~~

---

## 11. 开发顺序

### M1：ADM

- 实现 adm_enhancer.py；
- 接入现有定位缓存；
- 输出 20 例增强图；
- 确认病灶突出且背景仍保留。

### M2：ResNet

- 实现 ResNet-101 patch encoder；
- 导出单例和小批量 visual cache；
- 验证 108×256 token。

### M3：数据合同

- 实现 AdmEvidenceBatch；
- 正式路径移除 ROI/geometry；
- 实现 visual + semantic 拼接。

### M4：报告生成

- 8–16 例过拟合；
- 训练 B0；
- 训练 B1；
- 训练 F0。

### M5：论文实验

- 三种子 B0/B1/F0；
- O0 上限；
- null/swap 干预；
- 标准文本与临床指标。

---

## 12. 目标命令

代码完成后：

~~~powershell
$py = 'C:\Users\HP\.conda\envs\torch-gpu1\python.exe'

& $py scripts\step01_build_manifest.py --config configs\adm_radgenome_f0.yaml
& $py scripts\step02a_export_adm.py --config configs\adm_radgenome_f0.yaml
& $py scripts\step02b_export_resnet.py --config configs\adm_radgenome_f0.yaml
& $py scripts\step02c_export_unimed_semantic.py --config configs\adm_radgenome_f0.yaml
& $py scripts\step02d_validate_evidence.py --config configs\adm_radgenome_f0.yaml

& $py scripts\step03_train.py --config configs\adm_radgenome_b0.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_b1.yaml
& $py scripts\step03_train.py --config configs\adm_radgenome_f0.yaml

& $py scripts\step04_generate.py --checkpoint checkpoints\adm_f0_resnet_unimed\best.pt --split test --max-len 256 --out results\adm_f0\test
& $py scripts\step05_evaluate.py --run results\adm_f0\test --interventions --concept-deltas
~~~

---

## 13. 完成定义

首版完成必须满足：

> 给定一例医学图像，冻结异常检测器产生预测注意力图；注意力图增强原图后，由冻结 ResNet-101 提取视觉 patch tokens；冻结 UniMed-CLIP 产生疾病语义 tokens；两类 token 拼接后由小型 Transformer 生成 Findings 和 Impression。正式报告模型没有独立 ROI token 支路，也没有位置 token。

