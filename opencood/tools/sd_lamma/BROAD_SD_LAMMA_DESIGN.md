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

### 5.1 为什么需要虚拟接收方

Broadcast mask 不能直接使用当前 ego demand `D_i`，否则它会退化为原来的 `M_{j->ego}`。为了让 sender 估计“对周围潜在 receiver 普遍有价值的区域”，BROAD-SD-LAMMA 引入虚拟接收方 token `V_k`。

默认 `K=8`，对应 sender 周围八个典型方向：

```text
front
front-left
front-right
left
right
rear
rear-left
rear-right
```

每个 token 包含：

- 相对方向编码 `(dx, dy)`
- 归一化距离或固定半径 `dist`
- visibility / importance prior

### 5.2 fixed 与 learnable 模式

`VirtualReceiverAttention` 支持两种 token 模式：

- `fixed`：使用固定方向和 prior，不引入额外可学习参数。
- `learnable`：在固定 token 上增加一个小的 learnable delta，旧 checkpoint 可通过 `strict=False` 加载。

配置项：

```yaml
broadcast:
  num_virtual_receivers: 8
  virtual_receiver_mode: fixed
  method: vra
  use_vra: true
```

### 5.3 VRA 计算方式

实现中，每个 BEV cell 构造轻量 query：

```text
q(r) = [x_norm(r), y_norm(r), feature_energy(r), 1]
```

虚拟 receiver token 提供 key/value：

```text
k_v = [dx, dy, dist, prior]
value_v = prior
```

注意力输出为：

```text
A(r, k) = softmax(q(r) · k_k / temperature)
D_j^B(r) = normalize(sum_k A(r, k) * value_k)
```

最终 demand 与 sender feature energy 做轻量融合：

```text
D_j^B = normalize(0.55 * spatial_virtual_receiver_demand + 0.45 * feature_energy)
```

这个过程只读取 sender 的 `Z_j`，不读取当前 ego 的完整 demand map。

### 5.4 soft-OR fallback

如果 `method=soft_or`，或 VRA 关闭，或 VRA 执行失败且 `use_soft_or_fallback=true`，模块使用无需训练的规则版本：

```text
D_j^B(r) = normalize(0.50 * soft_or(direction_alignment(r, V_k)) + 0.50 * feature_energy(r))
```

这保证 broadcast 模式在没有额外训练权重时也能 forward。

## 6. Broadcast Utility Selection

每个 sender 的 broadcast utility 定义为：

```text
U_j(r) = S_j(r) * D_j^B(r) * R_j / C_j(r)
```

第一版中 `C_j(r)=1`，即所有 BEV cell 通信代价相同。

各项来源如下：

- `S_j`：sender supply map。实现上优先使用 `confidence_head` 得到 pre-fusion confidence；若失败则回退到 LAMMA BEV feature energy。
- `D_j^B`：由 `VirtualReceiverAttention` 或 soft-OR fallback 得到。
- `R_j`：Light-SAD modality reliability；若没有 Light-SAD 状态，则默认 1.0。
- `B_j`：sender-side broadcast budget，由现有 SD-LAMMA 网络预算逻辑估计。

对每个 scene，模块按 `record_len` 切分，避免跨 scene 混合。对每个非 ego sender 单独执行：

```text
candidate_j = supply_mask_j AND U_j > 0
M_j^B = TopK(U_j, candidate_j, budget_j)
P_j^B = Z_j * M_j^B
```

ego 自身特征保持不被 broadcast mask 裁剪。

## 7. Sender-side Budget 语义

原 SD-LAMMA 已有预算字段：

```yaml
network:
  budget_mode: "threshold"
  max_comm_ratio: null
  bandwidth_mbps: null
  latency_ms: null
  deadline_ms: null
  packet_loss: null
  frame_rate_hz: 10.0
```

BROAD-SD-LAMMA 复用这些字段，但语义改为“每个 sender 最多广播多少比例的 BEV 区域”。

若 `network.max_comm_ratio` 有值，则优先作为 sender-side ratio。若没有显式 ratio，但有 bandwidth / latency / deadline / packet loss，则用现有 `_effective_budget_ratio` 估计可发送比例。若两者都没有，则回退到：

```yaml
broadcast:
  budget_ratio: 0.3
```

因此 `--sd_lamma_max_comm_ratio 0.3` 在 broadcast 模式下表示每个 sender 最多广播 30% BEV cells。

## 8. Ego-side Receiver Gating

Broadcast sender message 一旦生成，不再随 receiver 改变。为了让当前 ego 更灵活地使用收到的广播特征，模块提供 receiver-side local gating：

```yaml
broadcast:
  receiver_gating:
    enabled: false
    source: uncertainty
    min_gate: 0.0
    strength: 1.0
```

当 `enabled=false`：

```text
P_j^B = Z_j * M_j^B
```

当 `enabled=true`：

```text
G_i = ego local uncertainty/confidence gate
P_{j->i}^{use} = P_j^B * warp(G_i, ego -> sender)
```

这里 gating 只影响当前 ego 使用收到的 dense feature，不影响 sender-side `M_j^B`、`sender_packet_count` 或 payload 统计。

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
