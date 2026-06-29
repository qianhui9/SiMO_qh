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

## 3. 核心模块设计思想（论文方法视角）

本模块的核心设计问题可以概括为：如何在只允许 sender 发送一份 broadcast message 的通信约束下，学习一个足够接近 receiver-specific pairwise SD-LAMMA 效果的 sender-side mask。原 pairwise SD-LAMMA 的优势在于它显式读取当前 ego 的 demand map，因此能生成面向当前 ego 的精确通信区域；但这种设计隐含了 receiver-conditioned unicast 假设，即同一个 sender 可以根据不同 receiver 生成不同 `M_{j->i}`。BROAD-SD-LAMMA 的论文动机则是把这一假设改写为 broadcast-compatible setting：sender 只能在自己的 BEV 坐标系中生成唯一的 `M_j^B`，该 mask 不随当前 ego 改变，后续任意 receiver 都只能接收同一份 sender-side payload。

从方法设计上，learnable VRA 蒸馏由三个互补组件组成：receiver distribution surrogate、pairwise teacher supervision 和 constrained broadcast optimization。三者分别回答“student 在不能看 ego demand 时用什么替代接收方需求”“怎样从已有 pairwise 模块中获得训练信号”“怎样防止 student 通过全选破坏通信预算”。

### 3.1 Receiver Distribution Surrogate：用结构化虚拟接收方近似潜在 receiver 分布

BROAD-SD-LAMMA 不直接把真实 ego demand 输入 sender mask，而是在 sender 端引入一组少量虚拟接收方 token `V = {v_k}`。这些 token 不是自由 embedding，而是由固定几何先验初始化，默认覆盖 sender 周围的 front、front-left、front-right、left、right、rear 等典型相对方向。论文上可以将其解释为一种 compact receiver distribution surrogate：真实交通场景中的潜在接收方位置和视野是动态的，单 ego evaluation 只观测其中一个接收方；VRA 用一组结构化 token 近似“周围典型接收者可能关心哪些区域”，从而在不读取当前 ego demand 的前提下，为 sender 构造 receiver-agnostic demand prior。

每个 BEV cell 的 query 由归一化坐标和 sender feature energy 组成：坐标项提供空间方位，feature energy 提供该区域是否有可发送语义内容的证据。VRA 将 cell query 与虚拟接收方 token 做 attention，得到 broadcast demand `D_j^B`。这使 student 不再是单纯的 feature-energy top-k，也不是 pairwise demand 的复制，而是一个受几何先验约束的 sender-side demand estimator。

learnable 模式只允许小幅 delta 修正：

```text
v_k = v_k^0 + tanh(delta_v_k) * alpha
```

这种参数化在论文中有两个含义。第一，它保留了可解释的 geometry prior，避免 token 变成无结构自由向量；第二，`delta_v_k=0` 时 learnable 与 fixed 近似一致，因此没有训练 checkpoint 时不会破坏 baseline 推理路径。temperature 和 prior scale 也被限制为轻量可学习参数，用于调整 attention sharpness 和方向 prior 强度，而不是重训练主干网络。

### 3.2 Pairwise Teacher Supervision：把 receiver-specific 最优性转化为 broadcast 覆盖目标

由于 broadcast student 不能使用 ego demand，直接用 detection loss 端到端训练会很慢且噪声较大。本实现利用已有 pairwise SD-LAMMA 作为 teacher：teacher 允许使用当前 ego demand 生成 `M_{j->ego}^T`，但该 mask 只作为监督信号，不进入 student forward。论文表述上，这相当于把 receiver-specific expert 的选择结果转化为 weak supervision，用来告诉 broadcast student：在当前场景中，哪些 sender-side 区域被一个强 receiver-conditioned 策略认为有价值。

这里没有要求 student 完全复制 teacher。原因是 teacher mask 是 ego-specific，而 student mask 必须 receiver-agnostic；如果强行做逐像素一致性，student 会被迫学习单一 ego 的偏置，反而违背 broadcast 语义。因此 coverage loss 采用非对称设计：重点惩罚 “teacher important but student not covered”，而不是惩罚 “student selected but teacher not selected”。这一设计对应论文中的核心取舍：broadcast mask 不追求对某个 ego 的最优稀疏子集，而追求在固定 sender budget 下覆盖 receiver-conditioned teacher 所揭示的高价值区域。

### 3.3 Constrained Broadcast Optimization：保持一发多收语义和通信预算

蒸馏训练中最容易出现的退化解是 student 为了覆盖 teacher 直接扩大 mask，甚至趋近全选。为避免这一点，BROAD-SD-LAMMA 把训练目标显式写成 constrained broadcast optimization：coverage loss 提升 teacher 区域覆盖，budget loss 约束 soft selected ratio 不超过 sender-side budget，invariance loss 约束 token 轻微扰动下的 broadcast demand 稳定性。

通信预算在 broadcast 模式中按 sender-side 定义：

```text
|M_j^B| / (H * W) <= B_j
```

这不同于 pairwise 模式中面向当前 ego 的 collaborator 总通信比例。无论场景中潜在 receiver 数量如何，一个 sender 在一帧中最多只发送一份 `M_j^B`。因此训练和 debug 都显式记录 `sender_packet_count` 和 `packets_per_sender_max`；后者应始终为 1，用于证明该方法没有退化成 per-receiver packet generation。

### 3.4 训练边界：轻量参数学习而非重训 SiMO 主干

为了让实验结论聚焦在 BROAD-SD-LAMMA 的 broadcast token 设计上，默认冻结 LiDAR encoder、Camera encoder、aligner、LAMMA、Pyramid Fusion、Detection Head、Light-SAD 和原 pairwise SD-LAMMA。optimizer 只更新 `VirtualReceiverAttention` 中的 token delta、temperature delta 和 prior scale 等轻量参数。这样做的论文意义是控制变量：性能变化主要来自 broadcast receiver surrogate 的学习，而不是主干表征能力变化。

这种分阶段训练也使实验叙事更清晰：fixed VRA 是无需训练的 geometry-prior broadcast baseline；learnable VRA 是利用 pairwise teacher 学到的数据驱动 broadcast prior；可选 detection fine-tuning 是第二阶段增强，而不是方法成立的必要条件。最终方法保留原 SiMO-PF dense feature interface，不改变 Pyramid Fusion 输入 shape，因此可以直接与原始 SiMO、pairwise SD-LAMMA、broadcast fixed VRA 做公平消融。

## 4. Teacher / Student 定义

Teacher 是原 pairwise SD-LAMMA：

```text
M_{j->ego}^T = SupplyDemandLAMMAComm(Z_j, D_ego, S_j)
```

Student 是 BROAD-SD-LAMMA：

```text
M_j^B = BroadcastSupplyDemandLAMMAComm(Z_j, S_j, V)
```

训练时 teacher 可以使用 ego demand 生成监督 mask，但 teacher 输出只进入 loss；student forward 不接收 ego demand，也不会为不同 receiver 生成不同 sender mask。每个 sender 每帧仍只有一份 `M_j^B`，debug 中 `packets_per_sender_max` 应始终为 1。

## 5. Loss 组成

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

## 6. Checkpoint 机制

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

## 7. Debug 指标

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

## 8. 推荐训练流程

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

即：
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py   --model_dir saved_models/SiMO-PF   --fusion_method intermediate   --sd_lamma_broadcast_enable   --sd_lamma_broadcast_method vra   --sd_lamma_virtual_receiver_mode learnable   --sd_lamma_learnable_ckpt opencood/logs/opv2v_lidarcamera_lamma3_pyramid_fusion_2026_06_27_11_04_36/broad_sd_lamma_learnable_latest.pth   --sd_lamma_max_comm_ratio 0.3   --sd_lamma_log
```

Step 5：可选 detection fine-tuning：

```bash
python opencood/tools/train_broad_sd_lamma_distill.py   --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml   --model_dir saved_models/SiMO-PF   --sd_lamma_learnable_ckpt opencood/logs/<run>/broad_sd_lamma_learnable_latest.pth   --broad_sd_use_detection_loss   --broad_sd_lambda_det 0.05   --broad_sd_lr 1e-4
```

## 9. 当前实现边界

- 仍使用 dense masked feature 输入 Pyramid Fusion，没有把通信表示替换成真实 sparse packet。
- teacher 只用于训练监督和 debug，不改变 broadcast 推理语义。
- student 不使用当前 ego 完整 demand map 作为 sender-side mask 输入。
- 默认冻结 LiDAR encoder、Camera encoder、aligner、LAMMA、Pyramid Fusion、Detection Head、Light-SAD 和原 pairwise SD-LAMMA。
- batch 内所有 teacher/student mask 都按 `record_len` 切分，避免跨 scene 混合。
