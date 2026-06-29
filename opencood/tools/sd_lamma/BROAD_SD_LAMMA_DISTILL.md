# BROAD-SD-LAMMA Learnable VRA 蒸馏训练说明

> 说明：本文档原本用于单独说明 `virtual_receiver_mode=learnable` 的 Pairwise Teacher 蒸馏训练。当前核心设计、论文式方法叙事、损失函数思想和训练边界已经统一合并到 `BROAD_SD_LAMMA_DESIGN.md`，尤其是第 8 节 “Learnable VRA 的 Pairwise Teacher 蒸馏训练设计”。后续撰写论文方法章节时，请以 `BROAD_SD_LAMMA_DESIGN.md` 作为唯一主文档。

本文档仅保留为历史入口和快速运行命令索引，避免旧链接失效。

## 1. 统一方法文档

请阅读：

```text
opencood/tools/sd_lamma/BROAD_SD_LAMMA_DESIGN.md
```

该文档已经整合以下内容：

- BROAD-SD-LAMMA 的 sender-side broadcast mask 设计；
- Virtual Receiver Attention 的 fixed / learnable token 参数化；
- Pairwise SD-LAMMA teacher export；
- 非对称 coverage loss、budget loss 与 invariance loss；
- 轻量 learnable checkpoint 保存与推理加载；
- 配置项、debug 字段和完整运行命令。

## 2. 快速运行命令

进入项目和环境：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

Learnable VRA 蒸馏 dry-run：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py \
  --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml \
  --model_dir saved_models/SiMO-PF \
  --sd_lamma_max_comm_ratio 0.3 \
  --broad_sd_dry_run \
  --broad_sd_max_train_iters 2 \
  --broad_sd_log_interval 1
```

正式 Pairwise Teacher 蒸馏：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py \
  --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml \
  --model_dir saved_models/SiMO-PF \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_learnable_alpha 0.1 \
  --broad_sd_lambda_cover 1.0 \
  --broad_sd_lambda_budget 0.1 \
  --broad_sd_lambda_inv 0.05
```

加载 learnable checkpoint 推理：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_broadcast_enable \
  --sd_lamma_broadcast_method vra \
  --sd_lamma_virtual_receiver_mode learnable \
  --sd_lamma_learnable_ckpt opencood/logs/<run>/broad_sd_lamma_learnable_latest.pth \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_log \
  --light_sad_max_batches 2
```
