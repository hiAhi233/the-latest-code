# 4090 24GB 云端训练包

本包包含 MedADM_RRG 源码、RadGenome 胶质瘤子集全部230例四序列 MRI、分割和英文报告，
以及真实预测定位缓存、UniMed语义缓存、修复后的v2视觉缓存（B0/B1各230例）。
这不是完整1007例 RadGenome；没有 MEN/MET/ISLES/WMH。

定位器、ResNet、UniMed 在本训练方案中均冻结。云端直接使用其预测缓存训练报告
Transformer，不需要这些上游模型权重，也不重新训练它们。原始导出代码保留在
external_tools；若更换影像、定位器或prompt并重算上游，须另行提供权重及依赖。
原有开发配置含Windows路径；云端请使用下面的cloud_run入口。

## 上传与环境

使用平台带 CUDA 的 PyTorch 镜像（Python>=3.10、torch>=2.3，以及匹配的torchvision）。
建议CPU至少8核、内存16GB以上、空闲磁盘15GB以上，解压到平台数据盘。

```bash
unzip MedADM_RRG_4090_GLI230_20261111.zip
cd MedADM_RRG
python scripts/verify_cloud_bundle.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
python -m pip install -r requirements-cloud.txt
python -m pip install -e . --no-deps
python scripts/cloud_run.py prepare
```

prepare按当前解压位置生成manifest，核对患者划分及全部证据缓存，默认161/23/46。
数据不需要再下载。BUNDLE_MANIFEST.json包含逐文件SHA256；压缩包另附整体SHA256。
如将数据另放目录，先 `export MEDADM_DATA_ROOT=/实际路径/BraTS_GLI`。

## 先测速，再训练

```bash
# 真实230例配置上跑3轮F0测速，不依赖B1初始化，产物单独存储
python scripts/cloud_run.py benchmark --epochs 3

# 最短主线：B1 -> F0，单随机种子，最多各100轮，支持验证集早停
python scripts/cloud_run.py train --experiments b1 f0 --seed 0 --evaluate

# 完整三组比较：B0原图、B1异常增强、F0增强+语义
python scripts/cloud_run.py train --experiments b0 b1 f0 --seed 0 --evaluate
```

保持 batch_size=8、grad_accum=2，有效batch=16。24GB不必盲目加大batch：只有161例
训练集，改变batch会减少每轮更新次数、改变实验条件。先测速再决定是否调参。
F0读取同seed的B1最佳模型，先冻结decoder 5轮再联合训练。
每次结果写入 checkpoints/cloud_b0_seed0 等及 results/cloud_b0_seed0/test。
重复运行同一输出目录默认拒绝覆盖；要重跑可以换seed或显式加 --overwrite。
当前训练器没有断点续训；best.pt/last.pt不含优化器和调度器状态，不能称为恢复训练。

## 评测与多种子

--evaluate会生成测试报告并计算文本/概念指标；默认不跑耗时的证据干预。
如需证据干预，在对应训练完成后运行：

```bash
python scripts/step05_evaluate.py --run results/cloud_f0_seed0/test --interventions --concept-deltas

# 其余种子各自从同seed的B1初始化，避免跨种子混用
python scripts/cloud_run.py train --experiments b0 b1 f0 --seed 1 --evaluate
python scripts/cloud_run.py train --experiments b0 b1 f0 --seed 2 --evaluate
```

此次打包不包含O0：O0需要使用GT单独生成定位、视觉及语义缓存，不能把预测缓存
重命名当Oracle使用。本包用于先完成B0/B1/F0主实验。
METEOR仍为meteor-lite；论文正式对比应统一评测工具和分词口径。

## 验证范围与时间

已在Windows torch-gpu1上检查实际数据、缓存与训练链路。Linux命令使用标准Python，
路径按解压位置解析；没有远程4090实测成绩。平台先运行benchmark，timing.json将
输出该机器上的每轮均耗时、100轮估计和峰值显存。测速包含启动、验证和checkpoint
写盘，因此短测速偏保守；早停则会缩短正式训练时间。
训练只更新小型报告Transformer与投影/embedding；三随机种子和消融不是单模型训练。

本地实测（2026-11-11）：RTX 5060 Laptop 8GB、batch=8、grad_accum=2，真实161例
训练/23例验证，F0从零训练3轮共15.84秒（包含初始化、验证、checkpoint写盘），
均摊5.28秒/轮，线性折算100轮约8.8分钟，PyTorch峰值allocated显存0.32GiB。
这是本地短测，不是4090成绩，不能直接按GPU算力倍数换算：NPZ解压/磁盘读取也占时间。
对4090平台建议预留：单F0 100轮约5–15分钟，B1+F0约10–30分钟，
B0/B1/F0单种子约15–45分钟；三种子约45–135分钟。均为待平台短测修正的估计，
不包含上传/环境安装、额外上游重算和耗时干预；报告生成与基础评价另留几分钟。
