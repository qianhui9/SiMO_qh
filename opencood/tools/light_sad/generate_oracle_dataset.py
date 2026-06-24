import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

try:
    from .feature_builder import LightSADFeatureBuilder
    from .sensor_stats import collect_light_sad_state, record_len_to_list, total_cavs
except ImportError:
    from opencood.tools.light_sad.feature_builder import LightSADFeatureBuilder
    from opencood.tools.light_sad.sensor_stats import collect_light_sad_state, record_len_to_list, total_cavs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline oracle labels for learned Light-SAD.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--per_cav", action="store_true")
    parser.add_argument("--actions", default="L,C,LC")
    parser.add_argument("--others_action", default="LC", choices=["L", "C", "LC"])
    parser.add_argument("--lambda_compute", type=float, default=0.0005)
    parser.add_argument("--lambda_comm", type=float, default=0.0005)
    parser.add_argument("--lambda_latency", type=float, default=0.0002)
    parser.add_argument("--compute_cost_l", type=float, default=1.0)
    parser.add_argument("--compute_cost_c", type=float, default=1.2)
    parser.add_argument("--comm_cost_l", type=float, default=1.0)
    parser.add_argument("--comm_cost_c", type=float, default=0.8)
    parser.add_argument("--lidar_payload_norm_voxels", type=float, default=10000.0,
                        help="Voxel count used to normalize LiDAR payload proxy into a bounded unit scale.")
    parser.add_argument("--min_payload_scale", type=float, default=0.25)
    parser.add_argument("--max_payload_scale", type=float, default=2.0)
    parser.add_argument("--latency_cost_l", type=float, default=1.0)
    parser.add_argument("--latency_cost_c", type=float, default=1.0)
    parser.add_argument("--sd_lamma_enable", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--range", default=None)
    return parser.parse_args()


def jsonable(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def clone_batch(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone_batch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_batch(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_batch(v) for v in obj)
    return copy.deepcopy(obj)


def configure_hypes(args):
    opt = argparse.Namespace(model_dir=args.model_dir, range=args.range, note="", fusion_method="intermediate")
    hypes = yaml_utils.load_yaml(args.hypes_yaml, opt) if args.hypes_yaml else yaml_utils.load_yaml(None, opt)
    if args.split == "train":
        hypes["validate_dir"] = hypes.get("root_dir", hypes.get("validate_dir"))
    elif args.split == "test":
        hypes["validate_dir"] = hypes.get("test_dir", hypes.get("validate_dir"))
    else:
        hypes["validate_dir"] = hypes.get("validate_dir", hypes.get("test_dir"))

    model_args = hypes["model"]["args"]
    light_sad = model_args.get("light_sad", {})
    model_args["light_sad"] = light_sad
    light_sad.update({
        "enabled": True,
        "policy": "force",
        "force_action": "LC",
        "force_actions": None,
        "per_cav": args.per_cav,
        "log": False,
        "safe_fallback": True,
    })
    if args.sd_lamma_enable:
        sd_cfg = model_args.get("sd_lamma", {})
        model_args["sd_lamma"] = sd_cfg
        sd_cfg["enabled"] = True
    return hypes


def set_force_action(model, action=None, actions=None):
    dispatcher = getattr(model, "light_sad", None)
    if dispatcher is None:
        raise RuntimeError("Model was not created with Light-SAD enabled.")
    dispatcher.cfg.policy = "force"
    dispatcher.cfg.force_action = action
    dispatcher.cfg.force_actions = actions


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def action_costs(action, state, args):
    lidar = state.get("lidar", {}) or {}
    camera = state.get("camera", {}) or {}
    network = state.get("network", {}) or {}
    l_on = "L" in action
    c_on = "C" in action
    compute = (args.compute_cost_l if l_on else 0.0) + (args.compute_cost_c if c_on else 0.0)

    # Keep payload proxies on a bounded unit scale. The previous raw
    # num_voxels/1000 term made LiDAR cost 5-10x larger than camera cost and
    # overwhelmed per-frame detection-loss differences, collapsing labels to C.
    lidar_scale = _clamp(
        float(lidar.get("num_voxels", 0.0)) / max(float(args.lidar_payload_norm_voxels), 1.0),
        args.min_payload_scale,
        args.max_payload_scale,
    )
    camera_scale = _clamp(
        1.0 + float(camera.get("contrast", 0.0)),
        args.min_payload_scale,
        args.max_payload_scale,
    )
    bandwidth = max(float(network.get("bandwidth_mbps", 1000.0) or 1000.0), 1.0e-3)
    rtt = max(float(network.get("rtt_ms", network.get("latency_ms", 0.0)) or 0.0), 0.0)
    queue_delay = max(float(network.get("queue_delay_ms", 0.0) or 0.0), 0.0)
    packet_loss = _clamp(float(network.get("packet_loss", 0.0) or 0.0), 0.0, 1.0)
    network_pressure = 1.0 + min(rtt / 100.0, 2.0) + min(queue_delay / 100.0, 2.0) + packet_loss
    if bandwidth < 10.0:
        network_pressure += min((10.0 / bandwidth) - 1.0, 4.0)

    lidar_payload = args.comm_cost_l * lidar_scale
    camera_payload = args.comm_cost_c * camera_scale
    comm = ((lidar_payload if l_on else 0.0) + (camera_payload if c_on else 0.0)) * network_pressure
    latency = ((args.latency_cost_l if l_on else 0.0) + (args.latency_cost_c if c_on else 0.0)) * network_pressure
    latency += comm / bandwidth
    return float(compute), float(comm), float(latency)


def quality_from_output(output, criterion, label_dict):
    if criterion is not None and label_dict is not None:
        loss = criterion(output, label_dict)
        return -float(loss.detach().cpu().item()), {"detection_loss": float(loss.detach().cpu().item())}
    cls_preds = output.get("cls_preds", None)
    if cls_preds is None:
        return 0.0, {"quality_proxy": 0.0}
    score = torch.sigmoid(cls_preds.detach()).flatten()
    if score.numel() == 0:
        return 0.0, {"quality_proxy": 0.0}
    topk = score.topk(k=min(100, score.numel())).values.mean()
    return float(topk.cpu().item()), {"quality_proxy": float(topk.cpu().item())}


def run_forced(model, batch_data, action, criterion, device, actions_string=None):
    if actions_string is not None:
        set_force_action(model, action=None, actions=actions_string)
    else:
        set_force_action(model, action=action, actions=None)
    local_batch = train_utils.to_device(clone_batch(batch_data), device)
    with torch.no_grad():
        output = model(local_batch["ego"])
        quality, details = quality_from_output(output, criterion, local_batch["ego"].get("label_dict", None))
    return quality, details, output


def load_done_keys(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((str(row.get("sample_id")), str(row.get("cav_id"))))
    return done


def make_record(sample_id, cav_id, state, feature_builder, utilities, qualities, compute, comm, latency, metadata):
    action_set = list(utilities.keys())
    label = max(action_set, key=lambda a: utilities[a])
    return {
        "sample_id": sample_id,
        "cav_id": cav_id,
        "is_ego": bool((state.get("cav", {}) or {}).get("is_ego", cav_id in {"ego", "0", 0})),
        "state_dict": jsonable(state),
        "feature_names": feature_builder.feature_names,
        "feature_vector": feature_builder.build_one(state, normalize=False).tolist(),
        "utility_L": float(utilities.get("L", 0.0)),
        "utility_C": float(utilities.get("C", 0.0)),
        "utility_LC": float(utilities.get("LC", 0.0)),
        "utility": {k: float(v) for k, v in utilities.items()},
        "label": label,
        "action_quality_dict": qualities,
        "compute_cost_dict": compute,
        "comm_cost_dict": comm,
        "latency_cost_dict": latency,
        "metadata": metadata,
    }


def main():
    args = parse_args()
    actions = [x.strip().upper() for x in args.actions.split(",") if x.strip()]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        output_path.unlink()
    done = load_done_keys(output_path) if args.resume else set()

    hypes = configure_hypes(args)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    print("[Light-SAD oracle] creating model")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    criterion = train_utils.create_loss(hypes) if "loss" in hypes else None
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    feature_builder = LightSADFeatureBuilder()
    action_distribution = {action: 0 for action in actions}
    quality_distribution = {action: 0 for action in actions}
    with output_path.open("a", encoding="utf-8") as f:
        for batch_idx, batch_data in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            if batch_data is None:
                continue
            base_ego = batch_data["ego"]
            record_len = base_ego.get("record_len", None)
            state = collect_light_sad_state(base_ego, record_len=record_len, per_cav=args.per_cav)
            total = total_cavs(record_len)
            sample_id = str(batch_idx)

            if args.per_cav:
                pc = state.get("per_cav", {})
                lengths = record_len_to_list(record_len)
                cav_states = []
                offset = 0
                for batch_local, length in enumerate(lengths or [total]):
                    for local_idx in range(length):
                        idx = offset + local_idx
                        cav_states.append({
                            "lidar": pc.get("lidar", [state.get("lidar", {})] * total)[idx],
                            "camera": pc.get("camera", [state.get("camera", {})] * total)[idx],
                            "network": pc.get("network", [state.get("network", {})] * total)[idx],
                            "history": state.get("history", {}),
                            "cav": {"index": idx, "is_ego": local_idx == 0, "num_cavs": length},
                        })
                    offset += length
                for cav_idx, cav_state in enumerate(cav_states):
                    key = (sample_id, str(cav_idx))
                    if key in done:
                        continue
                    utilities, qualities, compute, comm, latency = {}, {}, {}, {}, {}
                    for action in actions:
                        forced = [args.others_action for _ in range(total)]
                        forced[cav_idx] = action
                        quality, detail, _ = run_forced(model, batch_data, action, criterion, device, actions_string=",".join(forced))
                        c_compute, c_comm, c_latency = action_costs(action, cav_state, args)
                        utility = quality - args.lambda_compute * c_compute - args.lambda_comm * c_comm - args.lambda_latency * c_latency
                        utilities[action] = utility
                        detail = dict(detail)
                        detail["quality_score"] = float(quality)
                        qualities[action] = detail
                        compute[action] = c_compute
                        comm[action] = c_comm
                        latency[action] = c_latency
                    rec = make_record(sample_id, str(cav_idx), cav_state, feature_builder, utilities, qualities, compute, comm, latency, {"split": args.split, "batch_idx": batch_idx, "oracle_mode": "per_cav"})
                    action_distribution[rec["label"]] += 1
                    quality_distribution[max(actions, key=lambda a: qualities[a].get("quality_score", float("-inf")))] += 1
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            else:
                key = (sample_id, "frame")
                if key in done:
                    continue
                frame_state = {
                    "lidar": state.get("lidar", {}),
                    "camera": state.get("camera", {}),
                    "network": state.get("network", {}),
                    "history": state.get("history", {}),
                    "cav": {"index": 0, "is_ego": True, "num_cavs": total},
                }
                utilities, qualities, compute, comm, latency = {}, {}, {}, {}, {}
                for action in actions:
                    quality, detail, _ = run_forced(model, batch_data, action, criterion, device)
                    c_compute, c_comm, c_latency = action_costs(action, frame_state, args)
                    utility = quality - args.lambda_compute * c_compute - args.lambda_comm * c_comm - args.lambda_latency * c_latency
                    utilities[action] = utility
                    detail = dict(detail)
                    detail["quality_score"] = float(quality)
                    qualities[action] = detail
                    compute[action] = c_compute
                    comm[action] = c_comm
                    latency[action] = c_latency
                rec = make_record(sample_id, "frame", frame_state, feature_builder, utilities, qualities, compute, comm, latency, {"split": args.split, "batch_idx": batch_idx, "oracle_mode": "frame"})
                action_distribution[rec["label"]] += 1
                quality_distribution[max(actions, key=lambda a: qualities[a].get("quality_score", float("-inf")))] += 1
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            print("[Light-SAD oracle] batch=%d action_distribution=%s" % (batch_idx, action_distribution))

    summary = {
        "output_path": str(output_path),
        "action_distribution": action_distribution,
        "quality_best_distribution": quality_distribution,
        "actions": actions,
        "cost_weights": {
            "lambda_compute": args.lambda_compute,
            "lambda_comm": args.lambda_comm,
            "lambda_latency": args.lambda_latency,
            "lidar_payload_norm_voxels": args.lidar_payload_norm_voxels,
            "min_payload_scale": args.min_payload_scale,
            "max_payload_scale": args.max_payload_scale,
        },
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
