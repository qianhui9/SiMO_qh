# Light-SAD v1 实现说明

Light-SAD v1 是 SiMO_qh 中的 per-CAV runtime modality scheduling prototype。它不是完整 EMC2，也不是完整 MoME：本版本只借鉴 EMC2 的前置轻量模态调度思想，以及 MoME 的局部可靠性建模思想，不接入 OpenPCDet、不接入 DETR query decoder/AQR/MED，也不接入真实网络 trace、真实 RTT、真实带宽或 packet delay。

## 主要实现

新增和扩展代码集中在 `opencood/tools/light_sad/`：

- `config.py`：扩展 `LightSADConfig`，新增 `per_cav`、history/local reliability 开关、EMC2-style proxy 阈值、debug dump 配置和 `force_actions`。
- `sensor_stats.py`：从真实 `processed_lidar`、`image_inputs` 中提取 batch/global 与 per-CAV 统计。LiDAR 优先按 `voxel_coords[:, 0]` 作为 flattened CAV index 分组；如果只能得到 batch-level 或无法可靠切分，则 fallback 并广播统计。
- `light_sad.py`：实现 batch/global 与 per-CAV 兼容调度输出，支持 `force`、`emc2_rule`、`emc2_rule_history`、`emc2_rule_local`、`emc2_rule_full`。
- `runtime_mask.py`：保留 global action mask，同时支持 flattened per-CAV actions 生成 `[B, max_cav]` 的 camera/lidar runtime mask。
- `history.py`：新增 `HistoryConfidenceBuffer`，用上一帧检测 score 的均值、top-k 均值、检测数量和 stale 状态辅助下一帧调度。
- `local_reliability.py`：新增 coarse BEV local reliability map，只作为调度辅助和 debug summary，不改变 Pyramid Fusion 权重。
- `verify_light_sad.py`：扩展为不依赖数据集、权重和 GPU 的 Light-SAD v1 单元验证。

主模型 hook 位于 `opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py`：

- Light-SAD 未启用时，原始双分支路径保持不变。
- Light-SAD 启用时，forward 开头调用 dispatcher。
- 如果所有 CAV 都不需要 camera，则跳过 camera encoder/backbone/aligner。
- 如果所有 CAV 都不需要 LiDAR，则跳过 LiDAR encoder/backbone/aligner。
- mixed per-CAV 动作下，先运行双分支，再在 LAMMA 前用 per-CAV runtime mask 屏蔽不需要的模态。
- 不修改 Pyramid Fusion、Detection Head 或后处理核心逻辑。

Inference hook 位于 `opencood/tools/inference.py`：

- 新增 per-CAV、history、local reliability、policy、force_actions、debug dump CLI。
- 如果 `--light_sad_use_history` 开启，每个 batch forward 前注入上一帧 history state；post-processing 后用 `pred_score` 更新 `HistoryConfidenceBuffer`。
- 如果取不到 score，则优雅降级为 stale history，不中断推理。

## 运行指令

进入项目并切换环境：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

静态语法检查：

```bash
python -m py_compile opencood/tools/light_sad/*.py \
  opencood/models/fuse_modules/lamma.py \
  opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py \
  opencood/tools/inference.py
```

Light-SAD 单元验证：

```bash
python -m opencood.tools.light_sad.verify_light_sad
```

原始推理回归测试，不启用 Light-SAD：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_max_batches 2
```

全局强制 LiDAR-only：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action L \
  --light_sad_log \
  --light_sad_max_batches 2
```

全局强制 Camera-only：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action C \
  --light_sad_log \
  --light_sad_max_batches 2
```

全局强制 LC：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action LC \
  --light_sad_log \
  --light_sad_max_batches 2
```

per-CAV mixed action smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_per_cav \
  --light_sad_force_actions L,LC,C \
  --light_sad_log \
  --light_sad_max_batches 2
```

history/local reliability smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_per_cav \
  --light_sad_use_history \
  --light_sad_use_local_reliability \
  --light_sad_log \
  --light_sad_max_batches 5
```
