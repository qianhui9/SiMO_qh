# -*- coding: utf-8 -*-
"""Train learnable BROAD-SD-LAMMA virtual receiver tokens by pairwise teacher distillation."""

import argparse
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import tqdm
import yaml

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.sd_lamma import (
    find_broadcast_comm,
    freeze_except_broad_sd_lamma,
    load_broad_sd_lamma_checkpoint,
    load_broadcast_learnable_checkpoint_from_config,
    save_broad_sd_lamma_checkpoint,
)


def train_parser():
    parser = argparse.ArgumentParser(description="BROAD-SD-LAMMA pairwise teacher distillation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help="Training yaml file.")
    parser.add_argument("--model_dir", default="",
                        help="Base SiMO checkpoint directory loaded with strict=False.")
    parser.add_argument("--output_dir", default=None,
                        help="Optional output directory for lightweight VRA checkpoints.")
    parser.add_argument("--fusion_method", "-f", default="intermediate",
                        help="Kept for consistency with train.py; not used during distillation.")

    parser.add_argument("--sd_lamma_mode", default="broadcast", choices=["pairwise", "broadcast"])
    parser.add_argument("--sd_lamma_broadcast_method", default="vra", choices=["soft_or", "vra"])
    parser.add_argument("--sd_lamma_virtual_receiver_mode", default="learnable",
                        choices=["fixed", "learnable"])
    parser.add_argument("--sd_lamma_learnable_ckpt", default=None,
                        help="Optional lightweight VRA checkpoint to resume/init from.")
    parser.add_argument("--sd_lamma_num_virtual_receivers", type=int, default=None)
    parser.add_argument("--sd_lamma_learnable_alpha", type=float, default=None)
    parser.add_argument("--sd_lamma_max_comm_ratio", type=float, default=None)

    parser.add_argument("--broad_sd_distill_enable", action="store_true", default=True,
                        help="Enable pairwise teacher distillation in the broadcast module.")
    parser.add_argument("--broad_sd_teacher_mode", default="pairwise", choices=["pairwise"])
    parser.add_argument("--broad_sd_freeze_backbone", action="store_true", default=True,
                        help="Freeze all non-BROAD-SD-LAMMA learnable parameters.")
    parser.add_argument("--broad_sd_no_freeze_backbone", action="store_false",
                        dest="broad_sd_freeze_backbone",
                        help="Debug only: do not freeze the backbone before building optimizer.")
    parser.add_argument("--broad_sd_trainable_scope", default="virtual_receiver",
                        choices=["virtual_receiver", "vra", "broadcast", "broadcast_comm", "all"])
    parser.add_argument("--broad_sd_lambda_cover", type=float, default=None)
    parser.add_argument("--broad_sd_lambda_budget", type=float, default=None)
    parser.add_argument("--broad_sd_lambda_inv", type=float, default=None)
    parser.add_argument("--broad_sd_lambda_det", type=float, default=None)
    parser.add_argument("--broad_sd_use_detection_loss", action="store_true")
    parser.add_argument("--broad_sd_max_train_iters", type=int, default=None)
    parser.add_argument("--broad_sd_dry_run", action="store_true",
                        help="Run a tiny shape/loss/gradient check and save no epoch sweep.")
    parser.add_argument("--broad_sd_lr", type=float, default=None)
    parser.add_argument("--broad_sd_weight_decay", type=float, default=None)
    parser.add_argument("--broad_sd_num_workers", type=int, default=4)
    parser.add_argument("--broad_sd_log_interval", type=int, default=None)
    parser.add_argument("--broad_sd_save_interval", type=int, default=None)
    return parser.parse_args()


def _ensure_nested(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = root.get(key, None)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def apply_broad_sd_overrides(hypes: Dict[str, Any], opt) -> Dict[str, Any]:
    model_args = hypes["model"]["args"]
    sd_cfg = _ensure_nested(model_args, "sd_lamma")
    sd_cfg["enabled"] = True
    sd_cfg["mode"] = opt.sd_lamma_mode
    _ensure_nested(sd_cfg, "network")
    _ensure_nested(sd_cfg, "demand")
    _ensure_nested(sd_cfg, "supply")
    _ensure_nested(sd_cfg, "redundancy")
    _ensure_nested(sd_cfg, "mask")
    _ensure_nested(sd_cfg, "debug")
    broadcast_cfg = _ensure_nested(sd_cfg, "broadcast")
    broadcast_cfg["enabled"] = True
    broadcast_cfg["method"] = opt.sd_lamma_broadcast_method
    broadcast_cfg["use_vra"] = opt.sd_lamma_broadcast_method == "vra"
    broadcast_cfg["virtual_receiver_mode"] = opt.sd_lamma_virtual_receiver_mode
    broadcast_cfg.setdefault("num_virtual_receivers", 8)
    broadcast_cfg.setdefault("learnable_alpha", 0.1)
    broadcast_cfg.setdefault("learnable_temperature", True)
    broadcast_cfg.setdefault("learnable_prior_scale", True)
    broadcast_cfg.setdefault("budget_ratio", 0.3)
    if opt.sd_lamma_num_virtual_receivers is not None:
        broadcast_cfg["num_virtual_receivers"] = opt.sd_lamma_num_virtual_receivers
    if opt.sd_lamma_learnable_alpha is not None:
        broadcast_cfg["learnable_alpha"] = opt.sd_lamma_learnable_alpha
    if opt.sd_lamma_learnable_ckpt is not None:
        broadcast_cfg["learnable_ckpt"] = opt.sd_lamma_learnable_ckpt
    if opt.sd_lamma_max_comm_ratio is not None:
        sd_cfg["network"]["max_comm_ratio"] = opt.sd_lamma_max_comm_ratio
        sd_cfg["network"]["budget_mode"] = "topk"

    distill_cfg = _ensure_nested(broadcast_cfg, "distill")
    distill_cfg["enabled"] = bool(opt.broad_sd_distill_enable)
    distill_cfg["teacher_mode"] = opt.broad_sd_teacher_mode
    distill_cfg["freeze_backbone"] = bool(opt.broad_sd_freeze_backbone)
    distill_cfg["trainable_scope"] = opt.broad_sd_trainable_scope
    distill_cfg["use_detection_loss"] = bool(opt.broad_sd_use_detection_loss)
    if opt.broad_sd_lambda_cover is not None:
        distill_cfg["lambda_cover"] = opt.broad_sd_lambda_cover
    distill_cfg.setdefault("lambda_cover", 1.0)
    if opt.broad_sd_lambda_budget is not None:
        distill_cfg["lambda_budget"] = opt.broad_sd_lambda_budget
    distill_cfg.setdefault("lambda_budget", 0.1)
    if opt.broad_sd_lambda_inv is not None:
        distill_cfg["lambda_inv"] = opt.broad_sd_lambda_inv
    distill_cfg.setdefault("lambda_inv", 0.05)
    if opt.broad_sd_lambda_det is not None:
        distill_cfg["lambda_det"] = opt.broad_sd_lambda_det
    distill_cfg.setdefault("lambda_det", 0.0)
    distill_cfg.setdefault("teacher_score_type", "mask")
    distill_cfg.setdefault("student_score_type", "utility")
    distill_cfg.setdefault("hard_topk_for_loss", False)
    distill_cfg.setdefault("token_dropout", 0.0)
    distill_cfg.setdefault("token_noise_std", 0.0)
    if opt.broad_sd_max_train_iters is not None:
        distill_cfg["max_train_iters"] = opt.broad_sd_max_train_iters
    if opt.broad_sd_log_interval is not None:
        distill_cfg["log_interval"] = opt.broad_sd_log_interval
    distill_cfg.setdefault("log_interval", 20)
    if opt.broad_sd_save_interval is not None:
        distill_cfg["save_interval"] = opt.broad_sd_save_interval
    distill_cfg.setdefault("save_interval", 1)
    return hypes


def _tensor_to_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _collect_metrics(debug: Dict[str, Any], det_loss=None) -> Dict[str, float]:
    keys = [
        "loss_total",
        "loss_cover",
        "loss_budget",
        "loss_inv",
        "teacher_selected_ratio",
        "student_selected_ratio",
        "teacher_student_overlap",
        "teacher_coverage_by_student",
        "broadcast_budget_ratio",
        "estimated_broadcast_payload_kbits",
        "sender_packet_count",
        "packets_per_sender_max",
        "num_virtual_receivers",
        "learnable_delta_norm",
        "learnable_delta_max_abs",
    ]
    metrics = {key: _tensor_to_float(debug.get(key, None)) for key in keys}
    metrics["loss_det"] = _tensor_to_float(det_loss, 0.0)
    if "distill_loss_total" in debug:
        metrics["loss_total"] = _tensor_to_float(debug["distill_loss_total"])
    return metrics


def _write_config_snapshot(saved_path: str, hypes: Dict[str, Any]) -> None:
    os.makedirs(saved_path, exist_ok=True)
    with open(os.path.join(saved_path, "config.yaml"), "w") as outfile:
        yaml.dump(hypes, outfile)


def _build_optimizer(hypes: Dict[str, Any], params, opt):
    optimizer_cfg = hypes.get("optimizer", {})
    method_name = optimizer_cfg.get("core_method", "Adam")
    optimizer_cls = getattr(torch.optim, method_name, None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {method_name}")
    lr = opt.broad_sd_lr if opt.broad_sd_lr is not None else float(optimizer_cfg.get("lr", 2.0e-4))
    args = dict(optimizer_cfg.get("args", {}) or {})
    if opt.broad_sd_weight_decay is not None:
        args["weight_decay"] = opt.broad_sd_weight_decay
    return optimizer_cls(params, lr=lr, **args)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes = apply_broad_sd_overrides(hypes, opt)
    distill_cfg = hypes["model"]["args"]["sd_lamma"]["broadcast"].get("distill", {})
    if opt.broad_sd_dry_run and distill_cfg.get("max_train_iters", None) is None:
        distill_cfg["max_train_iters"] = 2

    print("Dataset Building")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    num_workers = 0 if opt.broad_sd_dry_run else int(opt.broad_sd_num_workers)
    loader_kwargs = {
        "batch_size": hypes["train_params"]["batch_size"],
        "num_workers": num_workers,
        "collate_fn": train_dataset.collate_batch_train,
        "shuffle": True,
        "pin_memory": True,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    print("Creating Model")
    model = train_utils.create_model(hypes)
    model = train_utils.load_pretrained_branches(hypes, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
        print(f"Loaded base checkpoint from {opt.model_dir}, epoch {init_epoch}.")
    else:
        init_epoch = 0

    if opt.sd_lamma_learnable_ckpt:
        load_broad_sd_lamma_checkpoint(opt.sd_lamma_learnable_ckpt, model, map_location="cpu", strict=False)
        print(f"Loaded BROAD-SD-LAMMA learnable checkpoint: {opt.sd_lamma_learnable_ckpt}")
    else:
        load_broadcast_learnable_checkpoint_from_config(model, map_location="cpu")

    comm = find_broadcast_comm(model)
    if comm is None:
        raise RuntimeError("BroadcastSupplyDemandLAMMAComm was not created. Check sd_lamma.mode=broadcast.")

    if opt.broad_sd_freeze_backbone:
        trainable_names = freeze_except_broad_sd_lamma(model, distill_cfg.get("trainable_scope", "virtual_receiver"))
    else:
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable BROAD-SD-LAMMA parameters found. Use virtual_receiver_mode=learnable.")
    print("Trainable BROAD-SD-LAMMA parameters:")
    for name in trainable_names:
        print(f"  - {name}")

    model.to(device)
    model.eval()
    comm.train()
    if hasattr(comm, "virtual_receiver"):
        comm.virtual_receiver.train()

    optimizer = _build_optimizer(hypes, trainable_params, opt)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=0)
    criterion = train_utils.create_loss(hypes) if bool(distill_cfg.get("use_detection_loss", False)) else None

    if opt.output_dir is not None:
        saved_path = opt.output_dir
        _write_config_snapshot(saved_path, hypes)
    else:
        saved_path = train_utils.setup_train(hypes)
    writer = SummaryWriter(saved_path)

    max_iters = distill_cfg.get("max_train_iters", None)
    max_iters = int(max_iters) if max_iters is not None else None
    log_interval = int(distill_cfg.get("log_interval", 20))
    save_interval = int(distill_cfg.get("save_interval", 1))
    lambda_det = float(distill_cfg.get("lambda_det", 0.0))
    epochs = 1 if opt.broad_sd_dry_run else int(hypes["train_params"].get("epoches", 1))
    global_iter = 0
    latest_metrics: Dict[str, float] = {}

    print("BROAD-SD-LAMMA distillation start")
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])
        pbar = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                pbar.update(1)
                continue
            if max_iters is not None and global_iter >= max_iters:
                break

            model.eval()
            comm.train()
            if hasattr(comm, "virtual_receiver"):
                comm.virtual_receiver.train()
            optimizer.zero_grad()
            model.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            output_dict = model(batch_data["ego"])
            debug = output_dict.get("sd_lamma_debug", {})
            distill_loss = debug.get("distill_loss_total", None)
            if distill_loss is None:
                raise RuntimeError("sd_lamma_debug has no distill_loss_total; check broadcast.distill.enabled.")

            final_loss = distill_loss
            det_loss = None
            if criterion is not None and lambda_det > 0.0:
                det_loss = criterion(output_dict, batch_data["ego"]["label_dict"]) * lambda_det
                final_loss = final_loss + det_loss

            final_loss.backward()
            optimizer.step()

            latest_metrics = _collect_metrics(debug, det_loss)
            latest_metrics["loss_total_with_det"] = _tensor_to_float(final_loss)
            for key, value in latest_metrics.items():
                writer.add_scalar(f"broad_sd_distill/{key}", value, global_iter)

            if global_iter % log_interval == 0:
                pbar.set_description(
                    "loss={:.4f} cover={:.4f} budget={:.4f} inv={:.4f} cov={:.4f} delta={:.6f}".format(
                        latest_metrics.get("loss_total_with_det", 0.0),
                        latest_metrics.get("loss_cover", 0.0),
                        latest_metrics.get("loss_budget", 0.0),
                        latest_metrics.get("loss_inv", 0.0),
                        latest_metrics.get("teacher_coverage_by_student", 0.0),
                        latest_metrics.get("learnable_delta_norm", 0.0),
                    )
                )
            global_iter += 1
            pbar.update(1)
        pbar.close()
        scheduler.step(epoch)

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(saved_path, f"broad_sd_lamma_learnable_epoch{epoch + 1}.pth")
            save_broad_sd_lamma_checkpoint(ckpt_path, model, epoch=epoch + 1,
                                           iteration=global_iter, stats=latest_metrics)
            latest_path = os.path.join(saved_path, "broad_sd_lamma_learnable_latest.pth")
            save_broad_sd_lamma_checkpoint(latest_path, model, epoch=epoch + 1,
                                           iteration=global_iter, stats=latest_metrics)
            print(f"Saved lightweight BROAD-SD-LAMMA checkpoint: {ckpt_path}")
        if max_iters is not None and global_iter >= max_iters:
            break

    writer.close()
    print("Training Finished, lightweight checkpoints saved to %s" % saved_path)
    print("Latest metrics:", latest_metrics)


if __name__ == "__main__":
    main()
