# -*- coding: utf-8 -*-
"""Distillation utilities for learnable BROAD-SD-LAMMA virtual receivers."""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _record_len_to_list(record_len) -> List[int]:
    if record_len is None:
        return []
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def _collab_weight(record_len, total: int, device, dtype) -> torch.Tensor:
    lengths = _record_len_to_list(record_len)
    if not lengths:
        lengths = [total]
    weight = torch.zeros((total, 1, 1, 1), device=device, dtype=dtype)
    offset = 0
    for cav_num in lengths:
        cav_num = int(cav_num)
        if cav_num > 1:
            weight[offset + 1: offset + cav_num].fill_(1.0)
        offset += cav_num
    if offset < total:
        weight[offset:].fill_(1.0)
    return weight


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _zero_like_score(score: torch.Tensor) -> torch.Tensor:
    return score.sum() * 0.0


class BroadcastDistillationLoss(nn.Module):
    """Pairwise-teacher loss for receiver-agnostic broadcast masks."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = cfg or {}
        self.lambda_cover = float(self.cfg.get("lambda_cover", 1.0))
        self.lambda_budget = float(self.cfg.get("lambda_budget", 0.1))
        self.lambda_inv = float(self.cfg.get("lambda_inv", 0.05))
        self.teacher_score_type = str(self.cfg.get("teacher_score_type", "mask")).lower()
        self.student_score_type = str(self.cfg.get("student_score_type", "utility")).lower()
        self.eps = float(self.cfg.get("eps", 1.0e-6))

    def forward(
        self,
        student: Dict[str, torch.Tensor],
        teacher: Optional[Dict[str, torch.Tensor]],
        record_len=None,
        budget_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_score = self._student_score(student).clamp(0.0, 1.0)
        device = student_score.device
        dtype = student_score.dtype
        collab = _collab_weight(record_len, student_score.shape[0], device, dtype)
        valid_cells = torch.clamp(collab.sum() * student_score.shape[-2] * student_score.shape[-1], min=1.0)

        teacher_score = None
        teacher_mask = None
        if teacher is not None:
            teacher_score = self._teacher_score(teacher, student_score).detach().clamp(0.0, 1.0)
            teacher_mask = self._teacher_mask(teacher, student_score).detach() > 0.25

        loss_cover = self._coverage_loss(student_score, teacher_score, collab)
        loss_budget, soft_selected_ratio = self._budget_loss(
            student_score, collab, valid_cells, budget_ratio
        )
        loss_inv = self._invariance_loss(student_score, student, collab, valid_cells)

        total = (
            self.lambda_cover * loss_cover
            + self.lambda_budget * loss_budget
            + self.lambda_inv * loss_inv
        )

        student_mask = student.get("broadcast_mask", None)
        if student_mask is None:
            student_mask = student_score > 0.5
        else:
            student_mask = _resize_like(student_mask.to(device=device, dtype=dtype), student_score) > 0.25
        metrics = self._metrics(
            total,
            loss_cover,
            loss_budget,
            loss_inv,
            student_score,
            student_mask,
            teacher_score,
            teacher_mask,
            collab,
            valid_cells,
            soft_selected_ratio,
        )
        return total, metrics

    def _student_score(self, student: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.student_score_type == "demand" and student.get("broadcast_demand") is not None:
            return student["broadcast_demand"]
        if self.student_score_type == "mask" and student.get("broadcast_mask") is not None:
            return student["broadcast_mask"].float()
        if student.get("broadcast_utility") is not None:
            return student["broadcast_utility"]
        if student.get("broadcast_demand") is not None:
            return student["broadcast_demand"]
        raise ValueError("Broadcast distillation requires student utility or demand tensor.")

    def _teacher_score(
        self,
        teacher: Dict[str, torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        key = "teacher_utility" if self.teacher_score_type in ("utility", "score") else "teacher_mask"
        score = teacher.get(key, None)
        if score is None:
            score = teacher.get("teacher_mask", None)
        if score is None:
            return torch.zeros_like(ref)
        return _resize_like(score.to(device=ref.device, dtype=ref.dtype), ref)

    def _teacher_mask(
        self,
        teacher: Dict[str, torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        mask = teacher.get("teacher_mask", None)
        if mask is None:
            mask = teacher.get("teacher_utility", None)
        if mask is None:
            return torch.zeros_like(ref)
        return _resize_like(mask.to(device=ref.device, dtype=ref.dtype), ref)

    def _coverage_loss(
        self,
        student_score: torch.Tensor,
        teacher_score: Optional[torch.Tensor],
        collab: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_score is None:
            return _zero_like_score(student_score)
        weighted_teacher = teacher_score * collab
        denom = weighted_teacher.sum()
        if float(denom.detach().item()) <= self.eps:
            return _zero_like_score(student_score)
        miss = F.relu(teacher_score - student_score) * weighted_teacher
        return miss.sum() / torch.clamp(denom, min=self.eps)

    def _budget_loss(
        self,
        student_score: torch.Tensor,
        collab: torch.Tensor,
        valid_cells: torch.Tensor,
        budget_ratio: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_ratio = (student_score * collab).sum() / valid_cells
        if budget_ratio is None:
            budget_ratio = self.cfg.get("budget_ratio", 0.3)
        target = torch.tensor(float(budget_ratio), device=student_score.device, dtype=student_score.dtype)
        target = torch.clamp(target, min=0.0, max=1.0)
        return F.relu(selected_ratio - target).pow(2), selected_ratio.detach()

    def _invariance_loss(
        self,
        student_score: torch.Tensor,
        student: Dict[str, torch.Tensor],
        collab: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        aug = student.get("broadcast_utility_aug", None)
        if aug is None:
            aug = student.get("broadcast_demand_aug", None)
        if aug is None:
            return _zero_like_score(student_score)
        aug = _resize_like(aug.to(device=student_score.device, dtype=student_score.dtype), student_score)
        return ((student_score - aug).pow(2) * collab).sum() / valid_cells

    def _metrics(
        self,
        total: torch.Tensor,
        loss_cover: torch.Tensor,
        loss_budget: torch.Tensor,
        loss_inv: torch.Tensor,
        student_score: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_score: Optional[torch.Tensor],
        teacher_mask: Optional[torch.Tensor],
        collab: torch.Tensor,
        valid_cells: torch.Tensor,
        soft_selected_ratio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hard_student_ratio = (student_mask.float() * collab).sum() / valid_cells
        metrics = {
            "loss_total": total,
            "loss_cover": loss_cover,
            "loss_budget": loss_budget,
            "loss_inv": loss_inv,
            "loss_det": _zero_like_score(student_score),
            "student_selected_ratio": hard_student_ratio.detach(),
            "student_soft_selected_ratio": soft_selected_ratio.detach(),
            "teacher_selected_ratio": _zero_like_score(student_score).detach(),
            "teacher_student_overlap": _zero_like_score(student_score).detach(),
            "teacher_coverage_by_student": _zero_like_score(student_score).detach(),
        }
        if teacher_score is None or teacher_mask is None:
            return metrics

        teacher_weight = teacher_score * collab
        teacher_denom = torch.clamp(teacher_weight.sum(), min=self.eps)
        teacher_cells = torch.clamp((teacher_mask.float() * collab).sum(), min=1.0)
        overlap = ((student_mask & teacher_mask).float() * collab).sum() / teacher_cells
        coverage = (student_score * teacher_weight).sum() / teacher_denom
        metrics.update({
            "teacher_selected_ratio": ((teacher_mask.float() * collab).sum() / valid_cells).detach(),
            "teacher_student_overlap": overlap.detach(),
            "teacher_coverage_by_student": coverage.detach(),
        })
        return metrics


def find_broadcast_comm(model: nn.Module) -> Optional[nn.Module]:
    for module in model.modules():
        if module.__class__.__name__ == "BroadcastSupplyDemandLAMMAComm":
            return module
    return None


def iter_broad_sd_lamma_trainable_params(
    model: nn.Module,
    trainable_scope: str = "virtual_receiver",
) -> Iterable[Tuple[str, nn.Parameter]]:
    comm = find_broadcast_comm(model)
    if comm is None:
        return []
    scope = str(trainable_scope or "virtual_receiver").lower()
    params = []
    for name, param in comm.named_parameters():
        if param is None:
            continue
        if scope in ("virtual_receiver", "vra", "learnable_token"):
            if name.startswith("virtual_receiver."):
                params.append(("sd_lamma_comm." + name, param))
        elif scope in ("broadcast", "broadcast_comm", "all"):
            params.append(("sd_lamma_comm." + name, param))
    return params


def freeze_except_broad_sd_lamma(
    model: nn.Module,
    trainable_scope: str = "virtual_receiver",
) -> List[str]:
    for param in model.parameters():
        param.requires_grad_(False)
    names = []
    for name, param in iter_broad_sd_lamma_trainable_params(model, trainable_scope):
        param.requires_grad_(True)
        names.append(name)
    return names


def broad_sd_lamma_learnable_state(model: nn.Module) -> Dict[str, Any]:
    comm = find_broadcast_comm(model)
    if comm is None:
        raise ValueError("BroadcastSupplyDemandLAMMAComm not found in model.")
    virtual_receiver = getattr(comm, "virtual_receiver", None)
    if virtual_receiver is None:
        raise ValueError("Broadcast comm has no virtual_receiver module.")
    vr_state = {
        key: value.detach().cpu()
        for key, value in virtual_receiver.state_dict().items()
    }
    comm_state = {
        key: value.detach().cpu()
        for key, value in comm.state_dict().items()
        if key.startswith("virtual_receiver.")
    }
    return {
        "virtual_receiver_state_dict": vr_state,
        "broadcast_comm_state_dict": comm_state,
        "config": {
            "num_virtual_receivers": int(getattr(comm, "num_virtual_receivers", 0)),
            "virtual_receiver_mode": getattr(virtual_receiver, "mode", "unknown"),
            "learnable_alpha": float(getattr(virtual_receiver, "learnable_alpha", 0.0)),
            "method": str(getattr(comm, "broadcast_method", "vra")),
        },
    }


def save_broad_sd_lamma_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int = 0,
    iteration: int = 0,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    state = broad_sd_lamma_learnable_state(model)
    state.update({
        "epoch": int(epoch),
        "iteration": int(iteration),
        "stats": stats or {},
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_broad_sd_lamma_checkpoint(
    path: str,
    model: nn.Module,
    map_location: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    comm = find_broadcast_comm(model)
    if comm is None:
        raise ValueError("BroadcastSupplyDemandLAMMAComm not found in model.")
    virtual_receiver = getattr(comm, "virtual_receiver", None)
    if virtual_receiver is None:
        raise ValueError("Broadcast comm has no virtual_receiver module.")

    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("virtual_receiver_state_dict", None)
    if state is None:
        state = checkpoint.get("state_dict", checkpoint)
    if any(str(key).startswith("virtual_receiver.") for key in state.keys()):
        state = {
            str(key).replace("virtual_receiver.", "", 1): value
            for key, value in state.items()
            if str(key).startswith("virtual_receiver.")
        }
    missing, unexpected = virtual_receiver.load_state_dict(state, strict=strict)
    return {
        "path": path,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint": checkpoint,
    }


def load_broadcast_learnable_checkpoint_from_config(
    model: nn.Module,
    map_location: str = "cpu",
) -> Optional[Dict[str, Any]]:
    comm = find_broadcast_comm(model)
    if comm is None:
        return None
    broadcast_cfg = getattr(comm, "broadcast_cfg", {}) or {}
    mode = str(broadcast_cfg.get("virtual_receiver_mode", "fixed")).lower()
    ckpt = broadcast_cfg.get("learnable_ckpt", None)
    if mode != "learnable":
        return None
    if not ckpt:
        print("[BROAD-SD-LAMMA] learnable mode has no learnable_ckpt; using fixed-like zero delta init.")
        return None
    if not os.path.exists(ckpt):
        print(f"[BROAD-SD-LAMMA] learnable_ckpt not found: {ckpt}; using fixed-like zero delta init.")
        return None
    info = load_broad_sd_lamma_checkpoint(ckpt, model, map_location=map_location, strict=False)
    print(f"[BROAD-SD-LAMMA] loaded learnable virtual receiver checkpoint: {ckpt}")
    return info
