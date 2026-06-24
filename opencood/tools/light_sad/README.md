# Light-SAD 实现说明

Light-SAD（Lightweight Scenario-Adaptive Modality Dispatcher）是在 SiMO 前端新增的 EMC2-style 轻量模态调度器。它在不先运行完整 LiDAR+Camera 双分支的前提下，根据当前 batch 的低成本状态选择 `L`、`C` 或 `LC`，再让被选模态继续经过原有 aligner 和 LAMMA，保证输出仍落在统一 BEV 语义空间。

## 主要实现

新增代码位于 `opencood/tools/light_sad/`：

- `config.py`：定义 `LightSADConfig`，支持从 YAML/CLI 注入的 dict 初始化，并忽略未知字段。
- `sensor_stats.py`：从 `processed_lidar`、`image_inputs` 和可选 `network_state` 中提取点数、体素数、亮度、对比度、blur proxy、带宽和 RTT 等轻量状态。
- `light_sad.py`：实现 `LightSADDispatcher`，按规则输出 batch-level 动作 `L`、`C` 或 `LC` 及原因。
- `runtime_mask.py`：把动作转成 LAMMA runtime mask，形状为 `[B, N]` 的 camera/lidar mask。
- `verify_light_sad.py`：不依赖数据集和权重的单元验证脚本。

模型侧 hook 保持默认关闭：

- `opencood/models/fuse_modules/lamma.py`：`LAMMA3.forward` 新增 `runtime_modality_mask=None`。未传 mask 时，原始 `single_mode` 和 `random_drop` 行为不变；传入 mask 时，在 embedding 后对 camera/lidar 分支做运行时屏蔽。
- `opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py`：读取 `model.args.light_sad`。启用后先调度，再按 batch-level 动作跳过未选模态的 encoder/backbone/aligner；LAMMA 前用同形状零特征补齐缺失模态，并传入 runtime mask。
- `opencood/tools/inference.py`：新增 Light-SAD CLI 参数，在创建模型前把配置注入 `hypes["model"]["args"]`。

第一版只做帧级/批级调度，当前 batch 内所有 CAV 共用同一个动作，避免改动 batch 组装和 per-CAV 动态分支逻辑。

## 运行指令

先进入项目并切换环境：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

静态语法检查：

```bash
python -m py_compile \
  opencood/tools/light_sad/config.py \
  opencood/tools/light_sad/sensor_stats.py \
  opencood/tools/light_sad/light_sad.py \
  opencood/tools/light_sad/runtime_mask.py \
  opencood/tools/light_sad/verify_light_sad.py \
  opencood/models/fuse_modules/lamma.py \
  opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py \
  opencood/tools/inference.py
```

调度器单元测试：

```bash
python -m opencood.tools.light_sad.verify_light_sad
```

不启用 Light-SAD 的原始推理回归测试：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
  --fusion_method intermediate \
  --light_sad_max_batches 2
```

强制 LiDAR-only 路径 smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action L \
  --light_sad_log \
  --light_sad_max_batches 2
```

强制 Camera-only 路径 smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action C \
  --light_sad_log \
  --light_sad_max_batches 2
```

强制 LiDAR+Camera 路径 smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_force_action LC \
  --light_sad_log \
  --light_sad_max_batches 2
```

自动调度路径 smoke test：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_log \
  --light_sad_max_batches 5
```
