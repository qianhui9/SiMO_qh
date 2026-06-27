# BROAD-SD-LAMMA 模块设计与实现说明

本文档说明当前仓库中 `BROAD-SD-LAMMA: Broadcast-compatible Supply-Demand LAMMA Communication` 的设计目标、模块结构、核心算法、代码接入方式、配置项、调试字段和运行命令。

## 1. 设计背景

原有 SD-LAMMA 是 receiver-specific / pair-wise 通信模块。它默认 batch 内 ego 车辆为 receiver，将 ego demand `D_i` 与协作车辆 supply `S_j` 结合，生成面向当前 ego 的定向 mask：

```text
M_{j->i} = f(D_i, S_j)
```

这种设计适合单 ego evaluation，但通信语义是“发送者为某个接收者定制消息”。在真实广播通信假设下，一个 sender 在一帧中更合理的行为是只广播一份消息，所有潜在 receiver 接收同一份 sender-side payload。因此新增 BROAD-SD-LAMMA，用 sender-side broadcast mask 表达：

```text
M_j^B = f(Z_j, S_j, V_k, R_j, B_j)
P_j^B = Z_j * M_j^B
```

其中 `M_j^B` 对 receiver 不变，当前 ego 只能在本地决定如何使用收到的广播特征，不能反向改变 sender 已经广播的消息。

## 2. 设计目标

BROAD-SD-LAMMA 遵守以下原则：

- 不重写 SiMO、Light-SAD、LAMMA 或 Pyramid Fusion 主干。
- 不删除、不替换原有 `SupplyDemandLAMMAComm`。
- 默认关闭，并且 `sd_lamma.mode` 缺省为 `pairwise`，保证老命令向后兼容。
- 仍保持单 ego evaluation，不改成 multi-ego evaluation。
- 通信模块仍接在 LAMMA 之后、Pyramid Fusion 之前。
- 输入仍是 LAMMA 后统一 BEV 特征 `Z_j`，不对 raw LiDAR、raw camera 或未对齐特征做 mask。
- broadcast sender mask 不直接使用当前 ego 的完整 demand map。
- 每个 sender 每帧只生成一份 broadcast mask `M_j^B`。
- Pyramid Fusion 输入 shape 不变，第一版仍使用 dense masked feature。

## 3. 新增文件与职责

新增文件如下：

```text
opencood/tools/sd_lamma/
├── broadcast_comm.py
├── virtual_receiver.py
└── BROAD_SD_LAMMA_DESIGN.md
```

`virtual_receiver.py` 定义 `VirtualReceiverAttention`，负责从 sender BEV feature 中估计 receiver-agnostic broadcast demand `D_j^B`。

`broadcast_comm.py` 定义 `BroadcastSupplyDemandLAMMAComm`，它继承 `SupplyDemandLAMMAComm`，复用原模块中已有的 confidence 构造、Light-SAD reliability 解析、预算估计、Top-K 选择、sparse debug 等工具，但改变 mask 生成语义：从 pair-wise receiver-conditioned mask 改为 sender-side broadcast mask。

`__init__.py` 导出：

```python
from .broadcast_comm import BroadcastSupplyDemandLAMMAComm
from .comm import SupplyDemandLAMMAComm
from .virtual_receiver import VirtualReceiverAttention
```

## 4. 整体数据流

BROAD-SD-LAMMA 的接入位置与原 SD-LAMMA 相同：

```text
LiDAR / Camera input
-> Light-SAD runtime modality scheduler
-> selected encoder / backbone / aligner
-> LAMMA runtime modality mask
-> unified BEV feature Z_j
-> BroadcastSupplyDemandLAMMAComm
-> dense masked broadcast feature P_j^B
-> PyramidFusion.forward_collab
-> detection head
```

输入保持与原 SD-LAMMA 接近：

```text
features: [sum(record_len), C, H, W]
record_len: 每个 scene 的 CAV 数量
affine_matrix: Pyramid Fusion 使用的 normalized pairwise transform
data_dict: batch 输入
light_sad_info: Light-SAD 调度状态和 reliability
runtime_modality_mask: LAMMA runtime modality mask
confidence_head: 可选 pre-fusion confidence head
```

输出保持兼容：

```text
masked_features: [sum(record_len), C, H, W]
sd_lamma_debug: dict
```

因此 Pyramid Fusion 和后续 detection head 不需要修改。

## 5. Virtual Receiver Attention

### 5.1 设计动机：从 receiver-specific demand 到 broadcast demand

在原始 pairwise SD-LAMMA 中，通信 mask 的生成依赖当前 ego 的需求图 `D_i`。这种形式可以写作：

```text
M_{j->i} = g(S_j, D_i)
```

其中 sender `j` 为 receiver `i` 定制发送内容。该建模方式在单 ego evaluation 中直接有效，但它隐含了一个较强假设：sender 知道当前 receiver 的完整需求，并且可以为不同 receiver 发送不同消息。这与无线广播通信中的常见物理约束并不一致。对于 broadcast-compatible 协同感知，更自然的目标是让 sender 在本地生成一份 receiver-agnostic message：

```text
M_j^B = g_B(Z_j, S_j, context_j)
```

这里 `M_j^B` 不应依赖某个具体 ego 的完整 demand map。BROAD-SD-LAMMA 因此引入 Virtual Receiver Attention，用一组少量虚拟接收方 token 近似 sender 周围潜在接收者的空间分布和通信需求先验。它的核心思想是：sender 不为某一个真实 receiver 定制消息，而是估计“哪些 BEV 区域对周围典型接收方具有普遍广播价值”。

这种设计在论文表述中可理解为一种 receiver distribution approximation：真实场景中潜在接收方的位置、朝向和可见性是动态且多样的，而单 ego evaluation 只观测其中一个 receiver。虚拟接收方 token 提供了一个固定或可学习的低维 surrogate set，用有限的方向先验覆盖潜在接收车辆的空间模式，从而避免直接使用 ego demand 带来的 receiver leakage。

### 5.2 虚拟接收方 token 的构造

默认设置 `K=8` 个虚拟接收方，对应 sender 周围八个典型相对方向：

```text
front, front-left, front-right, left, right, rear, rear-left, rear-right
```

每个 token `v_k` 由四类轻量状态组成：

```text
v_k = [dx_k, dy_k, rho_k, pi_k]
```

其中：

- `(dx_k, dy_k)` 表示虚拟 receiver 相对于 sender 的方向编码；
- `rho_k` 表示归一化距离或固定感知半径；
- `pi_k` 表示 visibility / importance prior，用于表达不同方向的默认通信价值；
- token 数量保持很小，避免引入明显推理开销。

当前实现提供两种 token 参数化方式：

```text
fixed:      v_k = v_k^0
learnable:  v_k = v_k^0 + tanh(Delta v_k) * alpha
```

`fixed` 模式完全由方向先验确定，适合无训练权重的直接推理和消融实验。`learnable` 模式在固定方向 token 上加入小幅可学习修正，使模型能够在训练或蒸馏阶段调整典型接收方的重要性，但仍受初始几何先验约束，不会退化成任意无结构 embedding。

### 5.3 BEV cell query 与 receiver-agnostic demand

对 sender `j` 的 LAMMA 后 BEV 特征 `Z_j in R^{C x H x W}`，每个 BEV cell `r` 构造一个轻量 query：

```text
q_j(r) = [x_norm(r), y_norm(r), e_j(r), 1]
```

其中 `x_norm(r)` 和 `y_norm(r)` 是归一化 BEV 坐标，`e_j(r)` 是 sender feature energy：

```text
e_j(r) = Normalize( mean_c |Z_j(c, r)| )
```

坐标项提供区域相对方位，能让虚拟 receiver token 表达前向、侧向、后向等空间偏好；feature energy 项提供 sender 端内容强度，使 demand 不仅依赖几何先验，也关注 LAMMA 对该区域提取到的语义响应。

虚拟接收方 attention 定义为：

```text
a_{j,k}(r) = softmax_k( q_j(r)^T v_k / tau )
d_j^V(r) = sum_k a_{j,k}(r) * pi_k
```

其中 `tau` 是温度系数，`pi_k` 是 token 的 importance prior。最后将虚拟接收方空间响应与 sender feature energy 融合，得到 broadcast demand：

```text
D_j^B(r) = Normalize( lambda_v * d_j^V(r) + (1 - lambda_v) * e_j(r) )
```

当前实现中 `lambda_v=0.55`。这一设计让 `D_j^B` 同时具备两类信息：一类是“潜在 receiver 可能关心哪些方向区域”的 broadcast prior，另一类是“sender 当前帧哪些区域确实有可发送内容”的 feature evidence。

从科研论文角度看，Virtual Receiver Attention 的价值在于它提供了一个介于两种极端之间的建模方式：

- 不像 pairwise SD-LAMMA 那样直接读取真实 ego demand；
- 也不像纯 feature-energy mask 那样完全忽略接收方分布；
- 而是用少量结构化 token 近似潜在 receiver distribution，得到 receiver-agnostic 但 communication-aware 的 demand map。

### 5.4 soft-OR fallback 的设计

为了保证 broadcast 模式不依赖额外训练权重，模块还提供 non-learned fallback。当 `method=soft_or`，或 `use_vra=false`，或 VRA 执行失败且 `use_soft_or_fallback=true` 时，使用方向先验的 soft-OR 聚合：

```text
d_j^{OR}(r) = 1 - product_k (1 - align(r, v_k) * pi_k)
D_j^B(r) = Normalize(0.50 * d_j^{OR}(r) + 0.50 * e_j(r))
```

soft-OR 的含义是：只要某个虚拟接收方方向认为该区域有广播价值，该区域的需求就会被提升；多个方向同时支持时需求进一步增强，但通过乘法补集形式保持数值范围稳定。这一 fallback 特别适合做无训练 smoke test 和论文 ablation：可以分离“broadcast 通信语义”本身与“VRA 可学习建模能力”的贡献。

## 6. Broadcast Utility Selection

### 6.1 设计目标

Virtual Receiver Attention 只回答“哪些区域具有潜在接收价值”，但实际通信还必须考虑 sender 是否在该区域有可靠信息，以及当前网络预算是否允许发送。Broadcast Utility Selection 将 demand、supply、modality reliability 和 budget 统一到一个 sender-side selection 问题中。

对 sender `j` 的每个 BEV cell `r`，broadcast utility 定义为：

```text
U_j(r) = S_j(r) * D_j^B(r) * R_j / C_j(r)
```

其中：

- `S_j(r)` 是 sender supply，衡量 sender 在 cell `r` 是否有值得发送的感知信息；
- `D_j^B(r)` 是 receiver-agnostic broadcast demand；
- `R_j` 是 Light-SAD 估计的 sender modality reliability；
- `C_j(r)` 是发送该 cell 的通信代价；当前第一版设为 `C_j(r)=1`。

该公式体现了一个乘性筛选思想：只有当区域同时“sender 有内容”“潜在 receiver 可能需要”“当前 modality 可靠”时，utility 才会高。乘性形式比加性形式更适合通信压缩，因为任一因素接近 0 都意味着该区域不适合作为稀缺通信资源的优先对象。

### 6.2 Supply、Demand 与 Reliability 的具体来源

`S_j(r)` 只从 LAMMA 后统一 BEV 特征估计，而不是从 raw LiDAR 或 raw camera 估计。实现中优先复用 Pyramid Fusion 的 `single_head_0` 作为 pre-fusion confidence proxy：

```text
S_j(r) = sigmoid(h_conf(Z_j))(r)
```

如果 confidence head 不可用，则回退到 feature energy：

```text
S_j(r) = Normalize(mean_c |Z_j(c, r)|)
```

这保证 broadcast selection 始终发生在跨模态对齐后的统一 BEV 空间中，符合“LAMMA 后、Pyramid Fusion 前”的模块定位。

`D_j^B(r)` 来自第 5 节的 VRA 或 soft-OR fallback。它不依赖当前 ego 的 demand，因此不会产生 receiver-specific mask。

`R_j` 来自 Light-SAD 输出的 per-CAV reliability。如果 Light-SAD 提供显式 `reliabilities`，模块直接使用；如果没有显式 reliability，则沿用原 SD-LAMMA 的状态估计逻辑，从 lidar/camera/history/local reliability 状态中估计；如果完全没有 Light-SAD 信息，则默认 `R_j=1`，保持原始路径兼容。

### 6.3 Sender-side Top-K 选择

给定 utility map 后，BROAD-SD-LAMMA 对每个 scene 按 `record_len` 切分，并对每个非 ego sender 独立执行 Top-K：

```text
candidate_j(r) = 1[S_j(r) >= theta_s] * 1[U_j(r) > 0]
M_j^B = TopK(U_j, candidate_j, ratio=B_j)
P_j^B = Z_j * M_j^B
```

其中 `B_j` 是 sender-side broadcast budget。与 pairwise SD-LAMMA 不同，这里的 Top-K 不在所有 collaborator 与 ego demand 的笛卡尔空间中竞争，而是每个 sender 独立决定自己要广播的 BEV cells。这样可以保证：

```text
每个 sender 每帧最多产生一份 M_j^B
```

并且该 mask 不随 receiver 改变。当前实现中 ego 自身特征保持完整，不受 broadcast mask 裁剪；非 ego sender 的 dense feature 被 `M_j^B` zero-mask 后送入 Pyramid Fusion。

### 6.4 与 pairwise redundancy filling 的区别

原 pairwise SD-LAMMA 中，redundancy-aware filling 在 ego 坐标系下比较不同 collaborator 对同一 ego demand cell 的收益，目标是减少多个 collaborator 对同一 receiver demand 的重复填充。Broadcast 模式不沿用这一逻辑，因为 broadcast message 的约束对象变成 sender 自身：sender 不知道也不应该为某个 ego 的 remaining demand 做定制化调度。

因此 BROAD-SD-LAMMA 的选择更接近一个 sender-local knapsack / Top-K 问题：在 sender 自己的 BEV 空间中，根据 broadcast utility 和 sender budget 选择有限区域。这样牺牲了一部分 receiver-specific 最优性，但换来了与广播通信假设一致的消息语义。

## 7. Sender-side Budget 语义

### 7.1 为什么 budget 必须从 receiver-side 改为 sender-side

在 pairwise SD-LAMMA 中，`network.max_comm_ratio` 可以理解为“协作车辆为当前 ego 发送的总区域比例”。但 broadcast 通信中，sender 不应为每个 receiver 单独分配一份预算，否则随着 receiver 数量增加，总发送量会线性增长，违背一发多收的广播假设。

因此 BROAD-SD-LAMMA 将预算定义为 sender-side constraint：

```text
|M_j^B| / (H * W) <= B_j
```

这表示每个 sender 在一帧中最多广播 `B_j` 比例的 BEV cells。无论后续有多少 receiver 接收该消息，sender-side payload 都不再增加。

### 7.2 显式比例预算

最直接的预算来自：

```yaml
network:
  max_comm_ratio: 0.3
```

在 broadcast 模式中，它表示：

```text
K_j = ceil(max_comm_ratio * H * W)
```

即每个 sender 最多选择 `K_j` 个 BEV cells。命令行：

```bash
--sd_lamma_max_comm_ratio 0.3
```

对应每个 sender 最多广播 30% 的 BEV 区域，而不是所有 sender-to-ego pair 的总比例。

### 7.3 网络状态推导预算

如果没有显式 `max_comm_ratio`，但配置或 runtime state 中提供了网络条件，则沿用原 SD-LAMMA 的 `_effective_budget_ratio` 逻辑，将 bandwidth、latency、deadline、packet loss 转换为可发送 payload：

```text
T_frame = 1 / frame_rate_hz
T_tx = min(T_frame, deadline_ms / 1000) - latency_ms / 1000
payload_bits = bandwidth_mbps * 1e6 * T_tx * (1 - packet_loss)
dense_bits_j = H * W * C * dtype_bits
B_j = clamp(payload_bits / dense_bits_j, 0, 1)
```

这样网络退化会自然降低每个 sender 的 broadcast ratio。例如 packet loss 增大、deadline 变短或 latency 增大时，可用 `payload_bits` 下降，Top-K 预算随之变小。

### 7.4 fallback budget

如果既没有显式 ratio，也没有网络状态，broadcast 模式回退到：

```yaml
broadcast:
  budget_ratio: 0.3
```

这是一个工程上安全的默认值：它让 broadcast 模式在最小配置下可运行，同时避免无预算时退化为全量 dense feature 广播。论文实验中可以将该值作为主要通信率控制变量，报告不同 budget ratio 下的精度-通信量折中。

### 7.5 payload 统计

Broadcast debug 中的 payload 统计按 sender-side selected cells 计算：

```text
estimated_broadcast_payload_bits = selected_cells * C * dtype_bits
estimated_broadcast_payload_kbits = estimated_broadcast_payload_bits / 1000
```

其中 `selected_cells` 只统计非 ego sender 的 `M_j^B` 中被选中的 cells。由于每个 sender 只产生一份 mask，debug 中同时记录：

```text
sender_packet_count
active_sender_packet_count
packets_per_sender_max = 1
```

这些字段用于证明 broadcast 模式的通信量不会按 receiver 数量重复计算。

## 8. Ego-side Receiver Gating

### 8.1 设计动机

Broadcast message 是 sender-side 一次性生成的，不能随当前 ego 改变；但在单 ego evaluation 中，当前 ego 对不同区域的需求仍然不同。如果完全不考虑 ego 状态，广播消息中某些区域可能对当前 ego 帮助有限。Ego-side Receiver Gating 的目标是在不破坏 broadcast 语义的前提下，让 ego 本地决定如何利用已收到的 `P_j^B`。

关键约束是：

```text
receiver gating 只能改变 ego 如何使用 P_j^B，不能改变 sender 已广播的 M_j^B。
```

因此它是 receive-side feature modulation，而不是 sender-side communication scheduling。

### 8.2 gating map 的构造

当前实现支持两种来源：

```yaml
broadcast:
  receiver_gating:
    source: uncertainty   # or confidence
```

若使用 uncertainty：

```text
G_i(r) = 1 - Conf_i(r)
```

表示 ego 对低置信区域更依赖协作信息。若使用 confidence：

```text
G_i(r) = Conf_i(r)
```

则更偏向保留 ego 自身认为可靠的区域响应。默认使用 uncertainty，因为它更符合协同感知中的补盲直觉：ego 不确定的区域更需要来自其他 CAV 的广播信息。

`Conf_i` 与 supply 一样，优先来自 pre-fusion confidence head；若不可用，则回退到 feature energy proxy。

### 8.3 坐标变换与作用位置

Pyramid Fusion 会在内部将各 CAV 特征 warp 到 ego 坐标系进行融合。为了不改变 Pyramid Fusion 接口，当前实现将 ego gating map 反向 warp 到 sender 特征所在坐标系，然后作用在已经 masked 的 sender dense feature 上：

```text
G_{i->j} = warp(G_i, ego -> sender)
P_{j->i}^{use} = (Z_j * M_j^B) * G_{i->j}
```

这里 `P_{j->i}^{use}` 是当前 ego 本地使用的特征版本，不是 sender 新发送的 packet。实际 sender payload 仍然是：

```text
P_j^B = Z_j * M_j^B
```

因此 gating 不计入 sender packet 数，也不改变 `estimated_broadcast_payload_kbits`。

### 8.4 gate 强度与下界

为了避免 gating 过强导致接收特征被完全抑制，配置提供两个稳定性参数：

```yaml
receiver_gating:
  min_gate: 0.0
  strength: 1.0
```

实现中先施加下界：

```text
G_prime = min_gate + (1 - min_gate) * G
```

再用 `strength` 控制 gating 强度：

```text
G_final = (1 - strength) + strength * G_prime
```

当 `strength=0` 时，receiver gating 等价于关闭；当 `strength=1` 时，完全使用 gating map；`min_gate>0` 则保证每个位置至少保留一定比例的接收特征。

### 8.5 论文中的语义边界

在论文表述中，Ego-side Receiver Gating 应被描述为 local selective fusion，而不是 communication mask generation。它解决的是“当前 ego 如何消费一份广播消息”的问题，而不是“sender 应该为当前 ego 发什么”的问题。

这一区分很重要：如果 gating 被错误地解释为 sender mask 的一部分，BROAD-SD-LAMMA 就会重新变成 receiver-specific 方法。当前实现通过 debug 统计和 payload accounting 明确区分二者：

- `broadcast_mask` / `communication_mask`：sender-side `M_j^B`；
- `receiver_gating_mask`：ego-side local modulation；
- `sender_packet_count` 和 payload 只统计 sender-side broadcast mask；
- receiver gating 不影响 `M_j^B`，不影响 packet count。

## 9. 与 pairwise SD-LAMMA 的模式切换

主模型接入位于：

```text
opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py
```

初始化逻辑为：

```python
self.sd_lamma_mode = str(sd_lamma_cfg.get("mode", "pairwise")).lower()
if self.sd_lamma_mode == "broadcast":
    self.sd_lamma_comm = BroadcastSupplyDemandLAMMAComm(sd_lamma_cfg)
elif self.sd_lamma_mode == "pairwise":
    self.sd_lamma_comm = SupplyDemandLAMMAComm(sd_lamma_cfg)
else:
    raise ValueError(...)
```

这样：

- 老配置无 `mode` 字段时，自动走 `pairwise`。
- `sd_lamma.enabled=false` 时，主模型不调用通信模块。
- `sd_lamma.enabled=true` 且 `mode=broadcast` 时，才启用 BROAD-SD-LAMMA。

## 10. CLI 覆盖

推理脚本新增参数位于：

```text
opencood/tools/inference.py
```

新增参数：

```text
--sd_lamma_mode {pairwise,broadcast}
--sd_lamma_broadcast_enable
--sd_lamma_broadcast_method {soft_or,vra}
--sd_lamma_num_virtual_receivers
--sd_lamma_receiver_gating
--sd_lamma_save_broadcast_debug
```

关键语义：

- `--sd_lamma_broadcast_enable` 会同时打开 `sd_lamma.enabled=true` 并设置 `mode=broadcast`。
- 只传 `--sd_lamma_mode broadcast` 不会自动打开 `enabled`，避免无意改变默认关闭路径。
- `--sd_lamma_broadcast_method soft_or` 会设置 `use_vra=false`。
- `--sd_lamma_save_broadcast_debug` 会保存 broadcast demand / mask / utility tensor，默认不保存大 tensor。

## 11. Debug 输出

Broadcast 模式下 `sd_lamma_debug` 重点字段包括：

```text
enabled
mode = broadcast
num_virtual_receivers
broadcast_method
broadcast_selected_ratio
broadcast_demand_mean
broadcast_supply_mean
broadcast_utility_mean
broadcast_budget_ratio
sender_packet_count
active_sender_packet_count
packets_per_sender_max
estimated_broadcast_payload_bits
estimated_broadcast_payload_kbits
receiver_gating_enabled
pairwise_teacher_overlap
```

其中最关键的是：

```text
sender_packet_count = scene 内非 ego sender 数
packets_per_sender_max = 1
```

这两个字段用于说明 broadcast 模式不是为不同 receiver 生成多份 `M_{j->i}`，而是每个 sender 每帧只生成一份 `M_j^B`。

若 `debug.save_masks=true` 或 `--sd_lamma_save_broadcast_debug`，会额外保存：

```text
broadcast_demand
broadcast_mask
broadcast_utility
communication_mask
demand_score
supply_score
multiscale_communication_mask
receiver_gating_mask
```

若 `mask.export_sparse=true`，则沿用原 SD-LAMMA 的 dense-to-sparse debug：

```text
sparse_indices
sparse_values
sparse_shape
```

第一版仍不把 Pyramid Fusion 输入替换为 sparse tensor，只用于检查未来 sparse serialization 的等价性。

## 12. Pairwise Teacher Overlap

Broadcast 模式提供可选 teacher overlap 统计，默认关闭。

开启方式：

```yaml
sd_lamma:
  broadcast:
    teacher_overlap: true
```

或：

```yaml
sd_lamma:
  debug:
    compare_pairwise_teacher: true
```

开启后，模块会临时调用原 `SupplyDemandLAMMAComm._mask_one_sample` 生成 pairwise teacher mask `M_{j->ego}^T`，再与 broadcast mask `M_j^B` 计算 overlap：

```text
pairwise_teacher_overlap = |M_j^B ∩ M_{j->ego}^T| / |M_{j->ego}^T|
```

这只是 debug / ablation 接口，不参与 loss，不强依赖 teacher，因此 teacher 关闭时 broadcast forward 不受影响。

## 13. 配置示例

默认配置已经加入：

```text
saved_models/SiMO-PF/config.yaml
opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml
```

推荐配置片段：

```yaml
sd_lamma:
  enabled: false
  mode: pairwise
  demand:
    use_lidar_density: true
    use_uncertainty: true
  supply:
    confidence_threshold: 0.01
    use_modality_reliability: true
  network:
    budget_mode: "threshold"
    max_comm_ratio: null
    bandwidth_mbps: null
    latency_ms: null
    deadline_ms: null
    packet_loss: null
    frame_rate_hz: 10.0
  broadcast:
    enabled: false
    method: vra
    num_virtual_receivers: 8
    virtual_receiver_mode: fixed
    use_vra: true
    use_soft_or_fallback: true
    budget_ratio: 0.3
    use_modality_reliability: true
    receiver_gating:
      enabled: false
      source: uncertainty
      min_gate: 0.0
      strength: 1.0
    debug:
      save_broadcast_demand: false
      save_broadcast_mask: false
      save_broadcast_utility: false
      save_receiver_gating: false
  mask:
    dense_zero_mask: true
    multiscale: true
    export_sparse: false
  debug:
    log: false
    save_masks: false
```

## 14. 最小运行命令

进入项目和环境：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

静态检查：

```bash
python -m py_compile \
  opencood/tools/sd_lamma/comm.py \
  opencood/tools/sd_lamma/virtual_receiver.py \
  opencood/tools/sd_lamma/broadcast_comm.py \
  opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py \
  opencood/tools/inference.py
```

原始 SiMO-PF 回归路径：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_max_batches 2
```

原始 pairwise SD-LAMMA：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_enable \
  --sd_lamma_mode pairwise \
  --sd_lamma_budget_mode topk \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_log \
  --light_sad_max_batches 2
```

Broadcast soft-OR fallback：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_broadcast_enable \
  --sd_lamma_broadcast_method soft_or \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_log \
  --light_sad_max_batches 2
```

Broadcast VRA：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --sd_lamma_broadcast_enable \
  --sd_lamma_broadcast_method vra \
  --sd_lamma_num_virtual_receivers 8 \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_log \
  --light_sad_max_batches 2
```

Light-SAD + Broadcast VRA + receiver gating：

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_enable \
  --light_sad_per_cav \
  --sd_lamma_broadcast_enable \
  --sd_lamma_broadcast_method vra \
  --sd_lamma_receiver_gating \
  --sd_lamma_max_comm_ratio 0.3 \
  --sd_lamma_log \
  --light_sad_max_batches 2
```

## 15. 实现边界

当前版本已经完成 broadcast-compatible 的主路径，但仍保持第一版工程边界：

- Pyramid Fusion 输入仍为 dense tensor，没有改成 sparse tensor。
- `C_j(r)` 暂设为 1.0，没有加入复杂区域代价模型。
- VRA 可 forward，但若要作为 learnable 模块充分发挥效果，后续仍需要训练或蒸馏。
- pairwise teacher overlap 仅作为 debug 统计，尚未扩展为训练 loss。
- 当前仍保持单 ego evaluation，broadcast 语义通过 sender-side mask 与 debug packet count 体现。

## 16. 一句话总结

BROAD-SD-LAMMA 在 LAMMA 统一 BEV 空间中，用虚拟接收方注意力估计 sender-side broadcast demand，并结合 sender supply、Light-SAD modality reliability 与 sender-side budget 生成唯一 broadcast mask `M_j^B`。当前 ego 可以本地 gating 接收后的 dense feature，但不能改变 sender 已广播的消息，从而在不破坏原始 SiMO-PF 和 pairwise SD-LAMMA 的前提下，引入符合广播通信假设的供需通信模式。
