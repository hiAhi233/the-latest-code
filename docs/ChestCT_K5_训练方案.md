# MedADM-RRG 胸部 CT 训练方案

在独立目录解压，等脑 MRI 任务结束后使用同一张 4090。流程依次运行 K=5 定位训练、全病例预测热图、视觉和语义缓存、报告 Transformer 训练、统一报告评分。

## 数据和监督

| 数据 | 用途 | 划分 |
|---|---|---|
| MSD Task06 Lung 的 CT 和肺肿瘤掩码 | 训练、校准和评测肺肿瘤定位头 | 5 例支持，5 例校准，5 例测试 |
| 本地 RadGenome-ChestCT 的 CT 和英文病例级长文本 | Transformer 训练及报告生成评测 | 按患者固定随机种子 0，70 / 10 / 20 |
| RadGenome-ChestCT 的解剖掩码和异常描述 | 随包保存，供定位结果核对和区域分析 | 100 例，各有 10 个解剖掩码 |

本地 100 例全部出自 RadGenome 官方验证集，因此此次是本地子集上的流程试验。每例采用区域报告表中 `Anatomy` 为空的单条病例级文本作为 Findings 目标，Impression 留空。正式报告实验可按相同 CT 文件名对齐 CT-RATE 的原始 Findings 和 Impression，并使用官方训练集。训练词表只从 70 例训练文本构建。

MSD-Lung 数据来自 [Medical Segmentation Decathlon](https://medicaldecathlon.com/dataaws/)，15 个原始 NIfTI 对取自[固定版本的文件镜像](https://huggingface.co/datasets/nielsRocholl/universal-lesion-segmentation/tree/c4c10c106ab3cbb6ac92aa6be895a3c52a6fce89/MSD/Task06_Lung)。固定病例名单见 `data/msd_lung/split.json`。解剖掩码覆盖肺、心脏等器官，肺肿瘤掩码描述的是肿瘤体素。实验得到的定位指标对应 MSD 的肺肿瘤范围。ChestCT 上保留预测热图用于复核，报告结果反映跨数据集定位迁移及 UniMed 语义的联合效果。

## 训练顺序

1. **K=5 定位。** 按固定的 5 个 MSD 患者采样。每例最多取 12 张有肿瘤标注的切片和 8 张同病例无标注肿瘤的切片。冻结 BiomedCLIP，随机初始化并训练原项目的 RecallDenseHead。AdamW 学习率 `1e-4`，600 步，每步 4 个含病灶视图和 4 个背景视图。用校准集的病例宏平均 Dice 选择阈值，用相同阈值评估 5 例留出测试。FP16 只用于 BiomedCLIP 推理，定位头用 FP32。
2. **在 4090 上生成热图。** 对 100 例逐层运行训练好的定位头。每例保留 3 个均匀分布的全局层和 3 个按热图分数挑选的层。同一组 6 层用于所有报告实验。定位使用肺窗，视觉和语义各读取肺窗与纵隔窗。两个窗的显示方向保持一致。
3. **提取 token 并训练报告 Transformer。** B0 将原图送入冻结的 ResNet-101。F0 将热图增强后的图送入同一个 ResNet-101，并将同批未增强图送入冻结的 UniMed-CLIP。每例 108 个视觉 token，F0 再加 9 个语义 token。两组 token 一起输入 2 层 Transformer encoder 和 4 层 decoder。B0 和 F0 用相同种子分别训练，最多 80 个 epoch，验证 loss 连续 12 个 epoch 未改善时早停。可增加 B1 来单独检验热图增强。
4. **同口径评测。** B0、F0 在相同 20 例上生成报告，用同一参考文本计算 BLEU-1 至 BLEU-4、ROUGE-L 和 METEOR 1.5。报告记录 `adm_applied`，方便核对增强实际覆盖了多少病例。只有在增强实际生效的病例中，才能看出热图是否帮助定位相关描述。

## 4090 执行命令

需要 Python 3.10 以上、可用的 CUDA PyTorch/torchvision（PyTorch 2.6 或以上）以及给 METEOR 使用的 Java。训练和缓存阶段从包内加载预训练权重。解压后进入 `MedADM_RRG_ChestCT`。

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pip install -r requirements-chest-ct.txt
java -version
python scripts/run_chest_ct.py --stage check
python scripts/run_chest_ct.py --stage localize
python scripts/run_chest_ct.py --stage report
python scripts/run_chest_ct.py --stage evaluate
```

也可以用 `python scripts/run_chest_ct.py` 依序运行全部阶段。要加 B1，在 report 和 evaluate 阶段使用同一组参数 `--experiments b0 b1 f0`。报告阶段可使用 `--epochs 2` 做快速试跑，正式试验用新解压目录保留各自结果。分阶段运行时，定位输出完成后可直接从 report 阶段继续。

| 输出 | 路径 |
|---|---|
| K=5 定位权重与留出测试 | `checkpoints/chestct_locator/checkpoint_selected.pt`、`msd_localization_metrics.json` |
| ChestCT 预测热图和三联图 | `caches/chest_ct/localization/arrays/`、`visualize/` |
| 生成和参考报告 | `results/chestct_{b0,f0}/test/predictions_test.jsonl` |
| 统一指标 | `results/chestct_eval_b0_f0/comparison.md` |

数据源 [RadGenome-ChestCT](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT) · [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
