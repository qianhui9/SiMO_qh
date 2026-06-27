# BROAD-SD-LAMMA Learnable VRA 蒸馏训练说明

本文档说明当前实现中 `virtual_receiver_mode=learnable` 的训练方式。目标是在不改变 sender-side broadcast 语义的前提下，用原 pairwise SD-LAMMA 作为 teacher，训练 BROAD-SD-LAMMA 的轻量 virtual receiver token / VRA 参数。

## 1. fixed 与 learnable 模式

`fixed` 模式使用固定几何方向先验 token：

```text
v_k = v_k^0
```

`learnable` 模式保留同一组固定几何先验，只允许小幅可学习修正：

```text
v_k = v_k^0 + tanh(delta_v_k) * alpha
```

当前默认 `alpha=0.1`，`delta_v_k` 初始化为 0，因此未加载 learnable checkpoint 时，learnable 模式会安全退化为接近 fixed 的行为。fixed 模式不注册 `token_delta`、`log_temperature_delta` 或 `prior_scale_delta` 这些可训练参数；learnable 模式会注册它们，optimizer 可以直接发现。

## 2. 为什么需要训练 learnable token

固定 virtual receiver token 只能表达人工设置的八方向先验。它能作为 broadcast baseline，但无法根据数据学习“哪些 sender-side BEV 区域更常被真实 ego pairwise demand 需要”。蒸馏训练让 VRA 在固定方向先验附近微调 token、temperature 和 prior scale，使 broadcast mask 更好覆盖 pairwise teacher 的重要区域，同时仍保持 receiver-agnostic。

## 3. Teacher / Student 定义

Teacher 是原 pairwise SD-LAMMA：

```text
M_{j->ego}^T = SupplyDemandLAMMAComm(Z_j, D_ego, S_j)
```

Student 是 BROAD-SD-LAMMA：

```text
M_j^B = BroadcastSupplyDemandLAMMAComm(Z_j, S_j, V)
```

训练时 teacher 可以使用 ego demand 生成监督 mask，但 teacher 输出只进入 loss；student forward 不接收 ego demand，也不会为不同 receiver 生成不同 sender mask。每个 sender 每帧仍只有一份 `M_j^B`，debug 中 `packets_per_sender_max` 应始终为 1。

## 4. Loss 组成

实现文件：`opencood/tools/sd_lamma/broadcast_distill.py`。

总损失为：

```text
L = lambda_cover * L_cover + lambda_budget * L_budget + lambda_inv * L_inv + lambda_det * L_det
```

- `L_cover`：惩罚 teacher 认为重要但 student utility 未覆盖的区域，默认使用 hard teacher mask + soft student utility。
- `L_budget`：约束 student soft selected ratio 不明显超过 sender-side budget。
- `L_inv`：可选 token dropout / noise 下的稳定性约束，默认权重较小。
- `L_det`：可选 detection loss 微调，默认 `lambda_det=0`，不参与第一阶段蒸馏。

teacher mask 由 `SupplyDemandLAMMAComm.export_pairwise_teacher()` 复用原 pairwise 逻辑导出，不另写一套不一致的 mask 生成器。

## 5. Checkpoint 机制

训练脚本只保存 BROAD-SD-LAMMA learnable 参数的轻量 checkpoint：

```text
broad_sd_lamma_learnable_epoch*.pth
broad_sd_lamma_learnable_latest.pth
```

保存内容包括：

- `virtual_receiver_state_dict`
- `broadcast_comm_state_dict` 中的 `virtual_receiver.*`
- `num_virtual_receivers`、`virtual_receiver_mode`、`learnable_alpha`、`method`
- epoch / iteration / loss 统计

推理时使用：

```yaml
sd_lamma:
  mode: broadcast
  broadcast:
    virtual_receiver_mode: learnable
    learnable_ckpt: path/to/broad_sd_lamma_learnable_latest.pth
```

或通过 CLI 覆盖：

```bash
--sd_lamma_virtual_receiver_mode learnable --sd_lamma_learnable_ckpt path/to/broad_sd_lamma_learnable_latest.pth
```

如果 learnable 模式未提供 checkpoint，系统会打印 fixed-like zero delta init 提示，并保持接近 fixed 的安全行为。

## 6. Debug 指标

蒸馏训练至少记录以下字段：

```text
loss_total
loss_cover
loss_budget
loss_inv
loss_det
teacher_selected_ratio
student_selected_ratio
teacher_student_overlap
teacher_coverage_by_student
broadcast_budget_ratio
estimated_broadcast_payload_kbits
sender_packet_count
packets_per_sender_max
num_virtual_receivers
learnable_delta_norm
learnable_delta_max_abs
virtual_receiver_mode
```

重点检查：

- `packets_per_sender_max == 1`：验证 broadcast 语义没有退化成 per-receiver packet。
- `learnable_delta_norm`：确认 token delta 确实被 optimizer 更新。
- `teacher_coverage_by_student`：观察 student 对 pairwise teacher 重要区域的覆盖。
- `student_selected_ratio` 和 `broadcast_budget_ratio`：确认 student 没有靠全选逃避 coverage loss。

## 7. 推荐训练流程

进入项目和环境：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

Step 1：验证 fixed broadcast 推理：

```bash
python opencood/tools/inference.py   --model_dir saved_models/SiMO-PF   --fusion_method intermediate   --sd_lamma_broadcast_enable   --sd_lamma_broadcast_method vra   --sd_lamma_virtual_receiver_mode fixed   --sd_lamma_max_comm_ratio 0.3   --sd_lamma_log   --light_sad_max_batches 2
```

Step 2：dry-run 蒸馏，检查 shape、loss 和可训练参数：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py   --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml   --model_dir saved_models/SiMO-PF   --sd_lamma_max_comm_ratio 0.3   --broad_sd_dry_run   --broad_sd_max_train_iters 2   --broad_sd_log_interval 1
```

Step 3：正式蒸馏训练，只训练 learnable VRA 轻量参数：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py   --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml   --model_dir saved_models/SiMO-PF   --sd_lamma_max_comm_ratio 0.3   --sd_lamma_learnable_alpha 0.1   --broad_sd_lambda_cover 1.0   --broad_sd_lambda_budget 0.1   --broad_sd_lambda_inv 0.05
```

Step 4：使用 learnable checkpoint 推理：

```bash
python opencood/tools/inference.py   --model_dir saved_models/SiMO-PF   --fusion_method intermediate   --sd_lamma_broadcast_enable   --sd_lamma_broadcast_method vra   --sd_lamma_virtual_receiver_mode learnable   --sd_lamma_learnable_ckpt opencood/logs/<run>/broad_sd_lamma_learnable_latest.pth   --sd_lamma_max_comm_ratio 0.3   --sd_lamma_log   --light_sad_max_batches 2
```

Step 5：可选 detection fine-tuning：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py   --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml   --model_dir saved_models/SiMO-PF   --sd_lamma_learnable_ckpt opencood/logs/<run>/broad_sd_lamma_learnable_latest.pth   --broad_sd_use_detection_loss   --broad_sd_lambda_det 0.05   --broad_sd_lr 1e-4
```

## 8. 当前实现边界

- 仍使用 dense masked feature 输入 Pyramid Fusion，没有把通信表示替换成真实 sparse packet。
- teacher 只用于训练监督和 debug，不改变 broadcast 推理语义。
- student 不使用当前 ego 完整 demand map 作为 sender-side mask 输入。
- 默认冻结 LiDAR encoder、Camera encoder、aligner、LAMMA、Pyramid Fusion、Detection Head、Light-SAD 和原 pairwise SD-LAMMA。
- batch 内所有 teacher/student mask 都按 `record_len` 切分，避免跨 scene 混合。
