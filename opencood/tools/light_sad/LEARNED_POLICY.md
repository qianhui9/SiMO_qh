# Learned Light-SAD 实现说明

本文档说明本项目中将规则式 Light-SAD 扩展为可训练轻量模态策略网络的主要实现。

## 主要思路

保留原始 `emc2_rule` 规则调度作为 baseline，不改变 SiMO、LAMMA、SD-LAMMA 和 Pyramid Fusion 的主干结构。新增 learned policy 只替换 Light-SAD 前端的动作选择逻辑：先用 `sensor_stats.py` 收集低成本状态，再由 `feature_builder.py` 转成固定维度向量，最后由 `learned_policy.py` 中的 MLP 输出 `L / C / LC` 三分类动作。

推理时 learned policy 只决定动作。被选中的模态仍会经过 encoder、backbone、aligner 和 LAMMA；未选中的模态在全局单模态场景会跳过对应分支，在 per-CAV mixed actions 下第一版采用“只要任一 CAV 需要该模态就运行分支，再用 runtime mask 屏蔽未选 CAV”的近似实现。

## 文件说明

- `feature_builder.py`：把 Light-SAD state dict 转成固定维度 tensor，支持 feature names、mean/std 归一化和 JSON 保存。
- `learned_policy.py`：轻量 MLP 策略网络，支持 checkpoint 保存/加载，checkpoint 内包含 action set、feature names、feature mean/std 和训练配置。
- `policy_dataset.py`：读取 oracle JSONL/JSON/PKL/PT 数据，返回 feature、hard label、soft utility、cost 和 metadata。
- `generate_oracle_dataset.py`：离线运行 `L / C / LC` 强制动作，通过同一 Light-SAD force action/runtime mask 路径生成 oracle 标签。
- `train_policy.py`：监督训练 MLP，支持 CE、utility KL distillation、expected cost penalty、class balance 和 utility margin 过滤。
- `eval_policy.py`：离线评估 policy，并可批量调用 `inference.py` 做接回 SiMO 的对照推理。
- `checkpoints/`：保存轻量策略网络 checkpoint 的默认目录。

## 策略模式

`light_sad.policy` 支持：

- `emc2_rule`：原始规则版调度。
- `learned_mlp`：加载 checkpoint 后直接用 MLP 预测动作。
- `hybrid`：优先使用 MLP；checkpoint 缺失、非法动作、低置信 margin、camera/lidar invalid 等情况回退到规则版或安全动作。

默认配置仍为 `enabled: false` 和 `policy: emc2_rule`，因此不启用 Light-SAD 时原始 SiMO-PF 路径保持不变。

## Oracle 数据生成

Frame-level 小样本：

```bash
cd /data/qh/phdCode/work3/SiMO_qh
conda activate SiMO_qh

CUDA_VISIBLE_DEVICES=0 python -m opencood.tools.light_sad.generate_oracle_dataset \
  --model_dir saved_models/SiMO-PF \
  --output_path saved_models/light_sad_policy/oracle_val_frame.jsonl \
  --split val \
  --max_batches 2
```

Per-CAV 小样本：

```bash
CUDA_VISIBLE_DEVICES=0 python -m opencood.tools.light_sad.generate_oracle_dataset \
  --model_dir saved_models/SiMO-PF \
  --output_path saved_models/light_sad_policy/oracle_val_per_cav.jsonl \
  --split val \
  --per_cav \
  --max_batches 2
```

## Policy 训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m opencood.tools.light_sad.train_policy \
  --train_path saved_models/light_sad_policy/oracle_train_per_cav.jsonl \
  --val_path saved_models/light_sad_policy/oracle_val_per_cav.jsonl \
  --save_dir saved_models/light_sad_policy \
  --hidden_dim 64 \
  --dropout 0.1 \
  --batch_size 256 \
  --lr 1e-3 \
  --epochs 20 \
  --alpha_kl 0.5 \
  --beta_cost 0.0 \
  --temperature 1.0 \
  --class_balance loss \
  --min_utility_margin 0.0
```

输出包括：

- `best.pth`
- `last.pth`
- `feature_norm.json`
- `train_log.json`
- `val_metrics.json`

## 离线评估

```bash
CUDA_VISIBLE_DEVICES=0 python -m opencood.tools.light_sad.eval_policy \
  --mode offline \
  --data_path saved_models/light_sad_policy/oracle_val_per_cav.jsonl \
  --checkpoint saved_models/light_sad_policy/best.pth \
  --output_dir saved_models/light_sad_policy/eval
```

## Learned Policy 推理

```bash
CUDA_VISIBLE_DEVICES=0 python -u opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --range 51.2,51.2 \
  --light_sad_enable \
  --light_sad_per_cav \
  --light_sad_policy learned_mlp \
  --light_sad_learned_ckpt saved_models/light_sad_policy/best.pth \
  --light_sad_feature_norm_path saved_models/light_sad_policy/feature_norm.json \
  --light_sad_temperature 1.0 \
  --light_sad_safe_fallback \
  --light_sad_log_policy_prob \
  --light_sad_log \
  --light_sad_dump_state \
  --light_sad_dump_path saved_models/SiMO-PF/light_sad_learned_debug.jsonl \
  --light_sad_max_batches 5
```

Hybrid + SD-LAMMA：

```bash
CUDA_VISIBLE_DEVICES=0 python -u opencood/tools/inference.py \
  --model_dir saved_models/SiMO-PF \
  --fusion_method intermediate \
  --range 51.2,51.2 \
  --light_sad_enable \
  --light_sad_per_cav \
  --light_sad_policy hybrid \
  --light_sad_learned_ckpt saved_models/light_sad_policy/best.pth \
  --light_sad_min_conf_margin 0.05 \
  --light_sad_safe_fallback \
  --light_sad_log_policy_prob \
  --sd_lamma_enable \
  --sd_lamma_budget_mode topk \
  --sd_lamma_max_comm_ratio 0.3
```

## 兼容性与限制

- `light_sad.enabled=false` 时不进入 Light-SAD 分支，保持原始 SiMO-PF 行为。
- `policy=emc2_rule` 时保留原规则版调度，可作为 learned policy baseline。
- checkpoint 缺失时默认 `safe_fallback=true`，会回退规则版；如果显式关闭 fallback，则会抛出清晰错误。
- per-CAV mixed action 的 encoder-level sparse execution 目前是近似实现：分支级别按 batch 是否需要该模态运行，CAV 级别由 runtime mask 屏蔽。
- 通信 payload、compute cost、latency cost 仍是估计值；后续可以接入真实网络 trace 或硬件 profiler。
- 当前 learned policy 是离线监督学习；RL fine-tuning 尚未接入。
