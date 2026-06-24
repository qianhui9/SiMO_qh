# SD-LAMMA / SD-SiMO 实现说明

本实现对应第二模块：Supply-Demand-Aware LAMMA Communication。它接在 SiMO aligner + LAMMA runtime mask 之后、Pyramid Fusion 之前，只对已经进入统一 BEV 语义空间的 LAMMA 输出特征做通信区域选择，不直接 mask LiDAR raw feature、Camera raw feature 或未对齐特征。

## 接入位置

主模型 hook 位于 `opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py`：

```text
LiDAR / Camera
-> Light-SAD runtime modality scheduler
-> selected encoder/backbone/aligner
-> LAMMA runtime modality mask
-> unified BEV feature Z_i
-> SupplyDemandLAMMAComm
-> dense masked collaborator BEV feature
-> PyramidFusion.forward_collab
-> detection head
```

`sd_lamma.enabled=false` 时不会调用 SD-LAMMA，模型保持原始 SiMO-PF 路径。`enabled=true` 时，ego 本车特征始终保留，mask 只作用于协作 CAV 发给 ego 的特征。

## 核心模块

核心实现文件：`opencood/tools/sd_lamma/comm.py`。

`SupplyDemandLAMMAComm` 的输入是 LAMMA 后的 `[sum(record_len), C, H, W]` BEV 特征、`record_len` 和 Pyramid Fusion 使用的 normalized `affine_matrix`。模块按 batch 内 ego 0 作为 receiver，逐样本生成 pair-wise `M_{j->i}`，避免跨 scene 混合。

### Demand D_i

Demand 是 soft score + binary mask：

- LiDAR density demand：从 `processed_lidar` 或 `inputs_m1` 的 `voxel_coords` / `voxel_num_points` 生成 BEV density map，低密度区域形成更高 demand。Camera-only 时会根据 Light-SAD runtime mask 跳过该项，不会把 demand 置零。
- Detection uncertainty demand：默认使用 Pyramid Fusion `single_head_0` 产生的 pre-fusion confidence；若 head 不可用，则回退到 LAMMA BEV feature energy。uncertainty 使用 `1 - confidence`。
- History demand：预留并支持读取 Light-SAD history state，默认关闭。

### Supply S_j

Supply 只从 LAMMA 后统一 BEV 特征生成，不从 raw modality 生成。默认优先复用 `pyramid_backbone.single_head_0` 作为轻量 pre-fusion confidence proxy；失败时回退到 feature energy。`supply.use_modality_reliability=true` 时会乘 Light-SAD/EMC2 reliability；Light-SAD 现在会输出 batch/per-CAV reliability，SD-LAMMA 也能在缺少显式字段时从 Light-SAD state 估计 reliability；若完全没有调度状态，则默认 reliability 为 1.0，保持向后兼容。

### Pair-wise Receiver-conditioned Mask

对每个 batch item，模块把 collaborator supply warp 到 ego 坐标系，与 ego demand 相乘：

```text
score_{j->i} = D_i * S_j * R_j
```

选择结果再用同一套 `affine_matrix` warp 回 sender 坐标系，对 `Z_j` 做 dense zero mask。这样 Pyramid Fusion 的输入 shape 不变，同时保留 receiver-conditioned 选择语义。

### Redundancy-aware Filling

`redundancy.enabled=true` 时，模块借鉴 CodeFilling 的 remaining-demand 去冗余思想：在 ego 坐标系中，对同一 BEV cell 只保留 gain 最高的 collaborator，低预算时通过 `network.max_comm_ratio` 继续 top-k 裁剪。这里不实现 codebook、vector quantization 或真实 sparse serialization。

### Debug 输出

启用 `debug.log=true` 或 CLI `--sd_lamma_log` 后，每个 batch 会打印：

- mean demand ratio
- mean supply ratio
- selected communication ratio
- redundancy 前 selected ratio
- estimated payload kbits
- per-modality selected ratio

`sd_lamma_debug` 会附加在模型 output dict 中；默认不保存 mask 张量，`debug.save_masks=true` 时才附加 demand/supply/communication mask 和 maxpool 版本的 multiscale mask。

## 配置

默认配置已加入：

- `saved_models/SiMO-PF/config.yaml`
- `opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml`

默认 `sd_lamma.enabled: false`，避免影响原始 SiMO 推理。

## 运行指令

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

静态检查：

```bash
python -m py_compile \
  opencood/tools/sd_lamma/comm.py \
  opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py \
  opencood/tools/inference.py
```

原始回归路径，SD-LAMMA 默认关闭：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_max_batches 2
```

启用普通 CoSDH-style pair-wise mask，不做去冗余：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_enable \
  --sd_lamma_no_redundancy \
  --sd_lamma_log \
  --sd_lamma_max_comm_ratio 0.3 \
  --light_sad_max_batches 2
```

启用 redundancy-aware filling：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_enable \
  --sd_lamma_log \
  --sd_lamma_budget_mode topk \
  --sd_lamma_max_comm_ratio 0.3 \
  --light_sad_max_batches 2
```

结合 Light-SAD per-CAV mixed modality：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_per_cav \
  --light_sad_force_actions L,LC,C \
  --light_sad_log \
  --sd_lamma_enable \
  --sd_lamma_log \
  --sd_lamma_budget_mode topk \
  --sd_lamma_max_comm_ratio 0.3 \
  --light_sad_max_batches 2
```
