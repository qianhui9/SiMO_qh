# BROAD-SD-LAMMA Implementation Notes

`BROAD-SD-LAMMA: Broadcast-compatible Supply-Demand LAMMA Communication`
is a parallel mode of the existing SD-LAMMA module. It does not replace
`SupplyDemandLAMMAComm` and does not rewrite SiMO, Light-SAD, LAMMA, or
Pyramid Fusion. It only inserts a sender-side broadcast mask after LAMMA has
produced unified BEV features and before `PyramidFusion.forward_collab`.

## Modes

- `sd_lamma.enabled: false`: original SiMO-PF path.
- `sd_lamma.enabled: true`, `sd_lamma.mode: pairwise`: original
  receiver-specific `M_{j->ego}` from `SupplyDemandLAMMAComm`.
- `sd_lamma.enabled: true`, `sd_lamma.mode: broadcast`: new
  `BroadcastSupplyDemandLAMMAComm`, with one `M_j^B` per sender and frame.

The default mode is `pairwise`, so old configs and commands stay compatible.

## New Files

- `virtual_receiver.py`: `VirtualReceiverAttention` estimates broadcast demand
  `D_j^B` from sender BEV feature energy and virtual receiver direction tokens.
- `broadcast_comm.py`: `BroadcastSupplyDemandLAMMAComm` reuses the existing
  confidence, Light-SAD reliability, budget, top-k, and sparse debug utilities
  while changing the mask semantics to sender-side broadcast.

First-version broadcast utility:

```text
U_j(r) = S_j(r) * D_j^B(r) * R_j
```

`C_j(r)` is kept as 1.0 for all cells.

## Receiver Gating

`broadcast.receiver_gating.enabled=false` sends `P_j^B = Z_j * M_j^B` directly
to Pyramid Fusion. When enabled, the current ego applies a local
uncertainty/confidence gate to the received dense feature. This gate never
changes the sender mask or sender-side payload accounting.

## Debug

Broadcast debug includes:

- `mode = broadcast`
- `broadcast_selected_ratio`
- `broadcast_demand_mean`
- `broadcast_supply_mean`
- `broadcast_utility_mean`
- `broadcast_budget_ratio`
- `num_virtual_receivers`
- `sender_packet_count`
- `estimated_broadcast_payload_kbits`
- `receiver_gating_enabled`
- `pairwise_teacher_overlap`

`sender_packet_count` counts non-ego senders per scene, proving the module
generates one packet per sender rather than one packet per receiver.

## Config

```yaml
sd_lamma:
  enabled: false
  mode: pairwise
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
```

Existing `network.max_comm_ratio`, `budget_mode`, `bandwidth_mbps`,
`latency_ms`, `deadline_ms`, and `packet_loss` are reused. In broadcast mode
they mean the maximum sender-side broadcast BEV ratio per sender.

## Commands

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh
```

Static check:

```bash
python -m py_compile \
  opencood/tools/sd_lamma/comm.py \
  opencood/tools/sd_lamma/virtual_receiver.py \
  opencood/tools/sd_lamma/broadcast_comm.py \
  opencood/models/point_pillar_lss_lamma2_pyramid_fusion.py \
  opencood/tools/inference.py
```

Original SiMO-PF regression path:

```bash
python opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --light_sad_max_batches 2
```

Original pairwise SD-LAMMA:

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

Broadcast soft-OR fallback:

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

Broadcast VRA:

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

Light-SAD + broadcast:

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
