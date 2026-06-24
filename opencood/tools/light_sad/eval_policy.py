import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .feature_builder import LightSADFeatureBuilder
    from .learned_policy import load_policy_checkpoint
    from .policy_dataset import OraclePolicyDataset, collate_policy_batch
    from .train_policy import evaluate
except ImportError:
    from opencood.tools.light_sad.feature_builder import LightSADFeatureBuilder
    from opencood.tools.light_sad.learned_policy import load_policy_checkpoint
    from opencood.tools.light_sad.policy_dataset import OraclePolicyDataset, collate_policy_batch
    from opencood.tools.light_sad.train_policy import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate learned Light-SAD policy.")
    parser.add_argument("--mode", default="offline", choices=["offline", "inference", "both"])
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_norm_path", default=None)
    parser.add_argument("--output_dir", default="saved_models/light_sad_policy/eval")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_dir", default="saved_models/SiMO-PF")
    parser.add_argument("--fusion_method", default="intermediate")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--sd_lamma_enable", action="store_true")
    return parser.parse_args()


def condition_tags(record):
    state = record.get("state_dict", record.get("state", {})) or {}
    camera = state.get("camera", {}) or {}
    lidar = state.get("lidar", {}) or {}
    network = state.get("network", {}) or {}
    tags = []
    if float(camera.get("dark_score", 0.0) or 0.0) >= 0.5:
        tags.append("night_or_dark")
    if float(camera.get("blur_proxy", 1.0) or 1.0) < 0.02:
        tags.append("camera_blur")
    if float(lidar.get("distant_sparse_score", 0.0) or 0.0) >= 0.2:
        tags.append("lidar_sparse")
    if float(network.get("bandwidth_mbps", 1000.0) or 1000.0) <= 5.0:
        tags.append("low_bandwidth")
    return tags or ["normal"]


@torch.no_grad()
def per_condition_metrics(model, dataset, device, action_set):
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_policy_batch, num_workers=0)
    for idx, batch in enumerate(loader):
        logits = model(batch["features"].to(device))
        pred = int(logits.argmax(dim=-1).cpu().item())
        label = int(batch["label"].item())
        utilities = batch["utilities"][0]
        regret = float(utilities.max().item() - utilities[pred].item())
        for tag in condition_tags(dataset.records[idx]):
            rows.append((tag, int(pred == label), regret))
    result = {}
    for tag in sorted(set(x[0] for x in rows)):
        subset = [x for x in rows if x[0] == tag]
        result[tag] = {
            "count": len(subset),
            "accuracy": float(sum(x[1] for x in subset) / max(len(subset), 1)),
            "utility_regret": float(sum(x[2] for x in subset) / max(len(subset), 1)),
        }
    return result


def run_inference_suite(args, output_dir):
    variants = [
        ("always_LC", ["--light_sad_enable", "--light_sad_force_action", "LC"]),
        ("always_L", ["--light_sad_enable", "--light_sad_force_action", "L"]),
        ("always_C", ["--light_sad_enable", "--light_sad_force_action", "C"]),
        ("rule", ["--light_sad_enable", "--light_sad_per_cav", "--light_sad_policy", "emc2_rule_full"]),
        (
            "learned",
            [
                "--light_sad_enable",
                "--light_sad_per_cav",
                "--light_sad_policy",
                "learned_mlp",
                "--light_sad_learned_ckpt",
                args.checkpoint,
                "--light_sad_log_policy_prob",
            ],
        ),
    ]
    if args.sd_lamma_enable:
        variants.append(
            (
                "learned_sd_lamma",
                [
                    "--light_sad_enable",
                    "--light_sad_per_cav",
                    "--light_sad_policy",
                    "learned_mlp",
                    "--light_sad_learned_ckpt",
                    args.checkpoint,
                    "--light_sad_log_policy_prob",
                    "--sd_lamma_enable",
                ],
            )
        )
    for name, extra in variants:
        cmd = [
            sys.executable,
            "-u",
            "opencood/tools/inference.py",
            "--model_dir",
            args.model_dir,
            "--fusion_method",
            args.fusion_method,
            "--save_vis_interval",
            "1000000",
            "--light_sad_dump_state",
            "--light_sad_dump_path",
            str(output_dir / ("%s_light_sad.jsonl" % name)),
        ] + extra
        if args.max_batches is not None:
            cmd += ["--light_sad_max_batches", str(args.max_batches)]
        print("[Light-SAD eval] running", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_policy_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    action_set = ckpt.get("action_set", model.action_set)

    if args.mode in {"offline", "both"}:
        if not args.data_path:
            raise ValueError("--data_path is required for offline evaluation.")
        if args.feature_norm_path:
            builder = LightSADFeatureBuilder.load_norm(args.feature_norm_path, feature_names=ckpt.get("feature_names"))
        else:
            builder = LightSADFeatureBuilder(
                ckpt.get("feature_names"),
                ckpt.get("feature_mean"),
                ckpt.get("feature_std"),
            )
        dataset = OraclePolicyDataset(args.data_path, action_set=action_set, feature_builder=builder, normalize=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_policy_batch, num_workers=0)
        metrics = evaluate(model, loader, device, action_set)
        metrics["per_condition"] = per_condition_metrics(model, dataset, device, action_set)
        (output_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))

    if args.mode in {"inference", "both"}:
        run_inference_suite(args, output_dir)


if __name__ == "__main__":
    main()
