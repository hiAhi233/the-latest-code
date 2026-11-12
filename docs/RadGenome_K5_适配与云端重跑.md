# RadGenome 胶质瘤 K=5 定位适配

2026-11-11 在本机 torch-gpu1 环境完成定位头训练与验证。支持集为固定随机种子 0
从训练集按患者抽取的 5 个病例，每例取 12 张病灶切片和 6 张有脑组织的无病灶切片。
5 个病例分别为 BraTS-GLI-00349-000、BraTS-GLI-00443-000、BraTS-GLI-00729-001、
BraTS-GLI-01069-000 和 BraTS-GLI-01313-000。训练、验证、测试患者集合已核验互斥。

从既有 MRI K5 权重初始化 RecallDenseHead，只更新该病灶定位头。
BiomedCLIP 和独立前景判别头保持冻结。训练沿用 recall_loss，学习率 1e-4，
每步混合 4 个病灶视图和 4 个负视图。第 200 和 600 步权重按验证集平均 Dice
选择，600 步胜出。测试使用固定的 600 步权重与旧阈值 0.2504556179。

| 指标 | 验证旧权重 | 验证适配后 | 测试旧权重 | 测试适配后 |
|---|---:|---:|---:|---:|
| 病例平均 Dice | 0.5353 | 0.6104 | 0.4916 | 0.5760 |
| 病灶召回率（连通域覆盖至少 10%） | 0.5100 | 0.6093 | 0.6286 | 0.7263 |
| 像素 AUROC | 0.9603 | 0.9724 | 0.9505 | 0.9699 |
| 热图峰值在病灶内 | 20/23 | 22/23 | 37/46 | 45/46 |
| Top-3 至少一层含病灶 | 20/23 | 23/23 | 40/46 | 46/46 |
| Top-3 三层平均病灶截面积 / 该例最大截面积 | 0.6534 | 0.7747 | 0.6854 | 0.8307 |
| 误报体积 / 真值病灶体积 | 0.4334 | 0.9538 | 0.4981 | 0.6257 |

这些是单一 5 病例支持集的结果。定位重叠、召回与切片选择改善，误报体积增加。
旧权重沿用既有实验的初始化，其原始训练患者与 RadGenome 的重合情况尚未核实。
因此本实验记录的是在既有权重基础上的目标域少样本适配。

适配后在本机用 FP16 编码器推理导出热图，定位头保持 FP32。
用第 200 步权重在 14 例上对照 FP32 和 FP16 导出，平均绝对热图误差约
0.00000714，阈值掩膜不一致比例约 0.00082%，13/14 例的 Top-3 切片一致。
旧模型指标读取上次实际使用的 FP32 缓存。评测使用同一 FLAIR、64×64 热图、
同一阈值和同一切片评分规则。

## 云端重跑

上传 K5 新包到服务器数据盘，在新目录解压，保留上轮 B0/B1/F0 结果。
新包的四类缓存来自同一份 600 步定位器。B0 和 B1 均使用新的 Top-3 切片，
B1 做 ADM 增强，F0 读取相同 ADM 视觉证据及在新切片上重算的 UniMed 语义。
本地导出配置放在 configs/radgenome_k5。打包把证据映射到现有云端配置读取的位置。
支持病例、选模记录及定位头权重在 localization_adaptation 目录。

```bash
unzip MedADM_RRG_4090_GLI230_K5Adapt_20261111.zip -d /root/autodl-tmp/medadm_k5
cd /root/autodl-tmp/medadm_k5/MedADM_RRG
python scripts/verify_cloud_bundle.py
python scripts/cloud_run.py prepare
python scripts/cloud_run.py train --experiments b0 b1 f0 --seed 0 --evaluate
```

使用与上次一致的 PyTorch 环境，按 CLOUD_README.md 安装依赖。
报告分数用 scripts/evaluate_standard_reports.py 在同一批 46 例 Findings 上比较。
在新解压目录中，用已有 EKAGen 评测依赖比较旧 F0 和新 F0。

```bash
python scripts/evaluate_standard_reports.py \
  --input Original_F0=/root/autodl-tmp/MedADM_RRG/results/cloud_f0_seed0/test/predictions_test.jsonl \
  --input K5_Adapted_F0=results/cloud_f0_seed0/test/predictions_test.jsonl \
  --scorer-root /root/autodl-tmp/EKAGen_RadGenome_4090_20261111/original \
  --section findings --out results/k5_vs_original_f0
```

结果包含 BLEU-1～4、METEOR 和 ROUGE-L。
