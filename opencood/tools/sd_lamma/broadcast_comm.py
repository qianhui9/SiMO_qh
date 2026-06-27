# -*- coding: utf-8 -*-
"""Broadcast-compatible supply-demand LAMMA communication."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from .comm import SupplyDemandLAMMAComm, _record_len_to_list
from .broadcast_distill import BroadcastDistillationLoss
from .virtual_receiver import VirtualReceiverAttention


class BroadcastSupplyDemandLAMMAComm(SupplyDemandLAMMAComm):
    """
    Sender-side broadcast SD-LAMMA scheduler.

    Unlike receiver-specific SD-LAMMA, this module generates one sender-side
    mask M_j^B per sender and frame. The current ego may apply local gating to
    the received dense feature, but that gate is not counted as a sender mask.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.mode = "broadcast"
        self.broadcast_cfg = self.cfg.get("broadcast", {}) or {}
        self.receiver_gating_cfg = self.broadcast_cfg.get("receiver_gating", {}) or {}
        self.broadcast_debug_cfg = self.broadcast_cfg.get("debug", {}) or {}

        self.num_virtual_receivers = int(self.broadcast_cfg.get("num_virtual_receivers", 8))
        self.num_virtual_receivers = max(1, self.num_virtual_receivers)
        self.broadcast_method = str(self.broadcast_cfg.get("method", "vra")).lower()
        self.virtual_receiver_mode = str(
            self.broadcast_cfg.get("virtual_receiver_mode", "fixed")
        ).lower()
        self.use_vra = bool(self.broadcast_cfg.get("use_vra", self.broadcast_method == "vra"))
        self.use_soft_or_fallback = bool(self.broadcast_cfg.get("use_soft_or_fallback", True))
        self.distill_cfg = self.broadcast_cfg.get("distill", {}) or {}
        self.distill_enabled = bool(self.distill_cfg.get("enabled", False))
        self.distill_loss = BroadcastDistillationLoss(self.distill_cfg) if self.distill_enabled else None
        self._distill_teacher_warned = False
        self.virtual_receiver = VirtualReceiverAttention(
            num_virtual_receivers=self.num_virtual_receivers,
            mode=self.virtual_receiver_mode,
            radius=float(self.broadcast_cfg.get("virtual_receiver_radius", 1.0)),
            temperature=float(self.broadcast_cfg.get("vra_temperature", 1.0)),
            learnable_alpha=float(self.broadcast_cfg.get("learnable_alpha", 0.1)),
            train_temperature=bool(self.broadcast_cfg.get("learnable_temperature", True)),
            train_prior_scale=bool(self.broadcast_cfg.get("learnable_prior_scale", True)),
            prior_scale_alpha=float(self.broadcast_cfg.get("prior_scale_alpha", 0.25)),
        )
        if self.virtual_receiver_mode == "learnable" and not self.broadcast_cfg.get("learnable_ckpt", None):
            print("[BROAD-SD-LAMMA] learnable virtual receiver uses fixed-like zero delta init.")

    def forward(
        self,
        features: torch.Tensor,
        record_len,
        affine_matrix: torch.Tensor,
        data_dict: Optional[Dict[str, Any]] = None,
        light_sad_info: Optional[Dict[str, Any]] = None,
        runtime_modality_mask: Optional[Dict[str, torch.Tensor]] = None,
        confidence_head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self.enabled or features is None:
            return features, {"enabled": False, "mode": "broadcast"}

        if features.dim() != 4:
            raise ValueError("BROAD-SD-LAMMA expects [sum_cav, C, H, W] features.")

        lengths = _record_len_to_list(record_len)
        if not lengths:
            lengths = [features.shape[0]]
        total_cavs = int(sum(lengths))
        if total_cavs != features.shape[0]:
            raise ValueError(
                "BROAD-SD-LAMMA record_len sum %d does not match features %d."
                % (total_cavs, features.shape[0])
            )

        self._active_network_state = self._extract_network_state(data_dict, light_sad_info)
        self._active_channels = int(features.shape[1])
        self._active_dtype_bits = self._dtype_bits(features.dtype)

        confidence = self._build_confidence(features, confidence_head)
        reliability = self._extract_reliability(
            light_sad_info, runtime_modality_mask, total_cavs, features.device, features.dtype
        )
        actions = self._extract_actions(light_sad_info, total_cavs)

        supply_score = confidence.clamp(0.0, 1.0)
        supply_mask = self._binary_from_score(
            supply_score,
            threshold=float(self.supply_cfg.get("confidence_threshold", 0.01)),
            topk_ratio=self.supply_cfg.get("topk_ratio", None),
        )
        broadcast_demand = self._build_broadcast_demand(features).to(
            device=features.device, dtype=features.dtype
        )
        reliability_factor = reliability if bool(
            self.broadcast_cfg.get("use_modality_reliability", True)
        ) else torch.ones_like(reliability)
        broadcast_utility = (supply_score * broadcast_demand * reliability_factor).clamp(0.0, 1.0)

        split_features = self._split(features, lengths)
        split_confidence = self._split(confidence, lengths)
        split_supply = self._split(supply_score, lengths)
        split_supply_mask = self._split(supply_mask.float(), lengths)
        split_demand = self._split(broadcast_demand, lengths)
        split_utility = self._split(broadcast_utility, lengths)

        out_features = []
        out_masks = []
        gating_masks = []
        sample_summaries = []
        offset = 0
        for batch_idx, cav_num in enumerate(lengths):
            sample_actions = actions[offset: offset + cav_num]
            sample_features, sample_mask, sample_gate, sample_debug = self._mask_one_sample_broadcast(
                split_features[batch_idx],
                split_confidence[batch_idx],
                split_supply[batch_idx],
                split_supply_mask[batch_idx],
                split_demand[batch_idx],
                split_utility[batch_idx],
                affine_matrix[batch_idx, :cav_num, :cav_num],
                sample_actions,
            )
            out_features.append(sample_features)
            out_masks.append(sample_mask)
            gating_masks.append(sample_gate)
            sample_summaries.append(sample_debug)
            offset += cav_num

        masked_features = torch.cat(out_features, dim=0)
        comm_mask = torch.cat(out_masks, dim=0)
        receiver_gating_mask = torch.cat(gating_masks, dim=0) if gating_masks else None

        debug = self._summarize_broadcast(
            sample_summaries,
            comm_mask,
            broadcast_demand,
            supply_score,
            broadcast_utility,
            features.shape[1],
            features.dtype,
            actions,
        )
        debug["modality_reliability"] = [
            float(x) for x in reliability.detach().view(-1).cpu().tolist()
        ]
        if hasattr(self.virtual_receiver, "learnable_stats"):
            debug.update(self.virtual_receiver.learnable_stats())

        distill_debug = self._maybe_compute_distill_loss(
            features,
            record_len,
            affine_matrix,
            data_dict,
            light_sad_info,
            runtime_modality_mask,
            confidence,
            reliability,
            actions,
            broadcast_demand,
            broadcast_utility,
            comm_mask,
            confidence_head,
        )
        if distill_debug is not None:
            debug.update(distill_debug)

        teacher_overlap = self._maybe_pairwise_teacher_overlap(
            features,
            record_len,
            affine_matrix,
            data_dict,
            light_sad_info,
            runtime_modality_mask,
            confidence,
            reliability,
            actions,
            comm_mask,
        )
        if teacher_overlap is not None:
            debug.update(teacher_overlap)
        else:
            debug["pairwise_teacher_overlap"] = None

        if self._should_save_debug_tensor("demand"):
            debug["broadcast_demand"] = broadcast_demand.detach().cpu()
        if self._should_save_debug_tensor("mask"):
            debug["broadcast_mask"] = comm_mask.detach().cpu()
        if self._should_save_debug_tensor("utility"):
            debug["broadcast_utility"] = broadcast_utility.detach().cpu()
        if receiver_gating_mask is not None and self._should_save_debug_tensor("receiver_gating"):
            debug["receiver_gating_mask"] = receiver_gating_mask.detach().cpu()
        if bool(self.debug_cfg.get("save_masks", False)):
            debug["communication_mask"] = comm_mask.detach().cpu()
            debug["demand_score"] = broadcast_demand.detach().cpu()
            debug["supply_score"] = supply_score.detach().cpu()
        if bool(self.debug_cfg.get("save_masks", False)) and bool(self.mask_cfg.get("multiscale", True)):
            debug["multiscale_communication_mask"] = self._make_multiscale_masks(comm_mask)

        if bool(self.mask_cfg.get("export_sparse", False)):
            debug.update(self._dense_to_sparse_debug(masked_features, comm_mask))

        self._maybe_log(debug)
        return masked_features, debug

    def _build_broadcast_demand(
        self,
        features: torch.Tensor,
        token_dropout: float = 0.0,
        token_noise_std: float = 0.0,
    ) -> torch.Tensor:
        method = str(self.broadcast_cfg.get("method", self.broadcast_method)).lower()
        use_vra = bool(self.broadcast_cfg.get("use_vra", self.use_vra))
        if method == "vra" and use_vra:
            try:
                return self.virtual_receiver(
                    features,
                    token_dropout=token_dropout,
                    token_noise_std=token_noise_std,
                )
            except Exception:
                if not self.use_soft_or_fallback:
                    raise
        return self.virtual_receiver.soft_or_demand(features)

    def _mask_one_sample_broadcast(
        self,
        features: torch.Tensor,
        confidence: torch.Tensor,
        supply_score: torch.Tensor,
        supply_mask: torch.Tensor,
        broadcast_demand: torch.Tensor,
        broadcast_utility: torch.Tensor,
        t_matrix: torch.Tensor,
        actions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cav_num, _, height, width = features.shape
        apply_mask = torch.ones((cav_num, 1, height, width), device=features.device, dtype=features.dtype)
        comm_mask = torch.zeros_like(apply_mask)
        receiver_gate = torch.ones_like(apply_mask)

        if cav_num <= 1:
            return features, comm_mask, receiver_gate, {
                "selected_ratio": 0.0,
                "broadcast_demand_mean": float(broadcast_demand.mean().item()) if cav_num else 0.0,
                "broadcast_supply_mean": 0.0,
                "broadcast_utility_mean": 0.0,
                "budget_ratio": self._broadcast_budget_ratio(height * width),
                "selected_cells": 0,
                "total_collab_cells": 0,
                "sender_packet_count": 0,
                "active_sender_packet_count": 0,
                "per_modality": {},
            }

        budget_ratio = self._broadcast_budget_ratio(height * width)
        budget = None
        if budget_ratio is not None:
            budget = int(torch.ceil(torch.tensor(float(height * width) * budget_ratio)).item())

        active_sender_packets = 0
        per_modality = defaultdict(lambda: {"selected": 0.0, "total": 0.0})
        for cav_idx in range(1, cav_num):
            candidate = (supply_mask[cav_idx:cav_idx + 1] > 0.25) & (
                broadcast_utility[cav_idx:cav_idx + 1] > 0.0
            )
            if budget is None:
                selected = candidate
            else:
                selected = self._topk_mask(
                    broadcast_utility[cav_idx:cav_idx + 1],
                    candidate,
                    budget,
                )
            apply_mask[cav_idx:cav_idx + 1] = selected.to(dtype=features.dtype)
            comm_mask[cav_idx:cav_idx + 1] = apply_mask[cav_idx:cav_idx + 1]
            if bool(selected.any().item()):
                active_sender_packets += 1

            mode = actions[cav_idx] if cav_idx < len(actions) else "LC"
            per_modality[mode]["selected"] += float(selected.float().sum().item())
            per_modality[mode]["total"] += float(height * width)

        masked_features = features * apply_mask
        if bool(self.receiver_gating_cfg.get("enabled", False)):
            receiver_gate = self._build_receiver_gating_sender(confidence, t_matrix)
            masked_features = masked_features.clone()
            masked_features[1:] = masked_features[1:] * receiver_gate[1:]

        selected_cells = int(comm_mask[1:].sum().item())
        total_collab_cells = int((cav_num - 1) * height * width)
        return masked_features, comm_mask, receiver_gate, {
            "selected_ratio": float(selected_cells / max(total_collab_cells, 1)),
            "broadcast_demand_mean": float(broadcast_demand[1:].mean().item()),
            "broadcast_supply_mean": float(supply_score[1:].mean().item()),
            "broadcast_utility_mean": float(broadcast_utility[1:].mean().item()),
            "budget_ratio": budget_ratio,
            "selected_cells": selected_cells,
            "total_collab_cells": total_collab_cells,
            "sender_packet_count": int(cav_num - 1),
            "active_sender_packet_count": int(active_sender_packets),
            "per_modality": dict(per_modality),
        }

    def _build_receiver_gating_sender(
        self,
        confidence: torch.Tensor,
        t_matrix: torch.Tensor,
    ) -> torch.Tensor:
        cav_num, _, height, width = confidence.shape
        source = str(self.receiver_gating_cfg.get("source", "uncertainty")).lower()
        if source == "confidence":
            ego_gate = confidence[0:1].clamp(0.0, 1.0)
        else:
            ego_gate = (1.0 - confidence[0:1]).clamp(0.0, 1.0)

        gate_in_ego = ego_gate.expand(cav_num, -1, -1, -1).contiguous()
        gate_sender = warp_affine_simple(
            gate_in_ego,
            t_matrix[:cav_num, 0],
            (height, width),
            align_corners=self.align_corners,
        ).clamp(0.0, 1.0)
        gate_sender[0].fill_(1.0)

        min_gate = max(0.0, min(1.0, float(self.receiver_gating_cfg.get("min_gate", 0.0))))
        strength = max(0.0, min(1.0, float(self.receiver_gating_cfg.get("strength", 1.0))))
        gate_sender = min_gate + (1.0 - min_gate) * gate_sender
        return (1.0 - strength) + strength * gate_sender

    def _broadcast_budget_ratio(self, sender_cells: int) -> Optional[float]:
        ratio = self._effective_budget_ratio(sender_cells)
        if ratio is None:
            ratio = self.broadcast_cfg.get("budget_ratio", 0.3)
        if ratio is None:
            return None
        return max(0.0, min(1.0, float(ratio)))

    def _summarize_broadcast(
        self,
        sample_summaries: List[Dict[str, Any]],
        comm_mask: torch.Tensor,
        broadcast_demand: torch.Tensor,
        supply_score: torch.Tensor,
        broadcast_utility: torch.Tensor,
        channels: int,
        dtype,
        actions: List[str],
    ) -> Dict[str, Any]:
        selected_cells = sum(int(x.get("selected_cells", 0)) for x in sample_summaries)
        total_collab_cells = sum(int(x.get("total_collab_cells", 0)) for x in sample_summaries)
        sender_packet_count = sum(int(x.get("sender_packet_count", 0)) for x in sample_summaries)
        active_sender_packet_count = sum(int(x.get("active_sender_packet_count", 0)) for x in sample_summaries)
        dtype_bits = self._dtype_bits(dtype)
        payload_bits = int(selected_cells * channels * dtype_bits)

        per_modality = defaultdict(lambda: {"selected": 0.0, "total": 0.0})
        for summary in sample_summaries:
            for mode, stats in summary.get("per_modality", {}).items():
                per_modality[mode]["selected"] += float(stats.get("selected", 0.0))
                per_modality[mode]["total"] += float(stats.get("total", 0.0))
        per_modality_ratio = {
            mode: stats["selected"] / max(stats["total"], 1.0)
            for mode, stats in per_modality.items()
        }

        def mean_value(key: str) -> float:
            vals = [x.get(key, None) for x in sample_summaries]
            vals = [float(x) for x in vals if x is not None]
            return float(sum(vals) / len(vals)) if vals else 0.0

        demand_mean = float(broadcast_demand[:, 0].detach().float().mean().item())
        supply_mean = float(supply_score[:, 0].detach().float().mean().item())
        utility_mean = float(broadcast_utility[:, 0].detach().float().mean().item())
        selected_ratio = float(selected_cells / max(total_collab_cells, 1))

        return {
            "enabled": True,
            "mode": "broadcast",
            "num_samples": len(sample_summaries),
            "num_cavs": int(comm_mask.shape[0]),
            "num_virtual_receivers": int(self.num_virtual_receivers),
            "broadcast_method": str(self.broadcast_cfg.get("method", self.broadcast_method)),
            "broadcast_selected_ratio": selected_ratio,
            "broadcast_demand_mean": demand_mean,
            "broadcast_supply_mean": supply_mean,
            "broadcast_utility_mean": utility_mean,
            "broadcast_budget_ratio": mean_value("budget_ratio"),
            "sender_packet_count": int(sender_packet_count),
            "active_sender_packet_count": int(active_sender_packet_count),
            "packets_per_sender_max": 1,
            "estimated_broadcast_payload_bits": payload_bits,
            "estimated_broadcast_payload_kbits": float(payload_bits / 1000.0),
            "receiver_gating_enabled": bool(self.receiver_gating_cfg.get("enabled", False)),
            "communication_rate": selected_ratio,
            "selected_cells": selected_cells,
            "total_collab_cells": total_collab_cells,
            "estimated_payload_bits": payload_bits,
            "estimated_payload_kbits": float(payload_bits / 1000.0),
            "mean_demand_ratio": demand_mean,
            "mean_supply_ratio": supply_mean,
            "mean_selected_ratio": selected_ratio,
            "mean_redundancy_before_ratio": 0.0,
            "per_modality_selected_ratio": per_modality_ratio,
            "actions": actions,
            "redundancy_enabled": False,
            "dense_zero_mask": bool(self.mask_cfg.get("dense_zero_mask", True)),
        }

    def _maybe_compute_distill_loss(
        self,
        features: torch.Tensor,
        record_len,
        affine_matrix: torch.Tensor,
        data_dict: Optional[Dict[str, Any]],
        light_sad_info: Optional[Dict[str, Any]],
        runtime_modality_mask: Optional[Dict[str, torch.Tensor]],
        confidence: torch.Tensor,
        reliability: torch.Tensor,
        actions: List[str],
        broadcast_demand: torch.Tensor,
        broadcast_utility: torch.Tensor,
        broadcast_mask: torch.Tensor,
        confidence_head: Optional[nn.Module],
    ) -> Optional[Dict[str, Any]]:
        if not self.distill_enabled or self.distill_loss is None:
            return None

        teacher_info = None
        try:
            with torch.no_grad():
                teacher_info = self.export_pairwise_teacher(
                    features,
                    record_len,
                    affine_matrix,
                    data_dict=data_dict,
                    light_sad_info=light_sad_info,
                    runtime_modality_mask=runtime_modality_mask,
                    confidence=confidence,
                    reliability=reliability,
                    actions=actions,
                    confidence_head=confidence_head,
                )
        except Exception as exc:
            if not self._distill_teacher_warned:
                print(f"[BROAD-SD-LAMMA] pairwise teacher export failed; using budget/invariance only: {exc}")
                self._distill_teacher_warned = True

        student = {
            "broadcast_demand": broadcast_demand,
            "broadcast_utility": broadcast_utility,
            "broadcast_mask": broadcast_mask,
        }
        lambda_inv = float(self.distill_cfg.get("lambda_inv", 0.0))
        token_dropout = float(self.distill_cfg.get("token_dropout", 0.0) or 0.0)
        token_noise_std = float(self.distill_cfg.get("token_noise_std", 0.0) or 0.0)
        if lambda_inv > 0.0 and (token_dropout > 0.0 or token_noise_std > 0.0):
            aug_demand = self._build_broadcast_demand(
                features,
                token_dropout=token_dropout,
                token_noise_std=token_noise_std,
            ).to(device=features.device, dtype=features.dtype)
            student["broadcast_demand_aug"] = aug_demand
            utility_scale = (
                broadcast_utility / torch.clamp(broadcast_demand.detach(), min=1.0e-6)
            ).detach()
            student["broadcast_utility_aug"] = (utility_scale * aug_demand).clamp(0.0, 1.0)

        budget_ratio = self._broadcast_budget_ratio(features.shape[-2] * features.shape[-1])
        loss, metrics = self.distill_loss(
            student,
            teacher_info,
            record_len=record_len,
            budget_ratio=budget_ratio,
        )
        debug: Dict[str, Any] = {
            "distill_enabled": True,
            "distill_teacher_available": teacher_info is not None,
            "distill_loss_total": loss,
            "broadcast_budget_ratio": float(budget_ratio) if budget_ratio is not None else 0.0,
        }
        for key, value in metrics.items():
            debug[key] = value
            debug[f"distill_{key}"] = value
        if teacher_info is not None:
            debug["pairwise_teacher_selected_cells"] = int(
                teacher_info.get("teacher_selected_cells", 0)
            )
        return debug

    def _maybe_pairwise_teacher_overlap(
        self,
        features: torch.Tensor,
        record_len,
        affine_matrix: torch.Tensor,
        data_dict: Optional[Dict[str, Any]],
        light_sad_info: Optional[Dict[str, Any]],
        runtime_modality_mask: Optional[Dict[str, torch.Tensor]],
        confidence: torch.Tensor,
        reliability: torch.Tensor,
        actions: List[str],
        broadcast_mask: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        enabled = bool(self.broadcast_cfg.get("teacher_overlap", False)) or bool(
            self.debug_cfg.get("compare_pairwise_teacher", False)
        )
        if not enabled:
            return None

        lidar_active = self._extract_lidar_active(runtime_modality_mask, features.shape[0], features.device)
        demand_score = self._build_demand_score(
            features, confidence, record_len, data_dict, lidar_active, light_sad_info
        )
        demand_mask = self._binary_from_score(
            demand_score,
            threshold=float(self.demand_cfg.get("uncertainty_threshold", 0.5)),
            topk_ratio=self.demand_cfg.get("topk_ratio", None),
        )
        teacher_supply = self._build_supply_score(confidence, reliability)
        teacher_supply_mask = self._binary_from_score(
            teacher_supply,
            threshold=float(self.supply_cfg.get("confidence_threshold", 0.01)),
            topk_ratio=self.supply_cfg.get("topk_ratio", None),
        )

        lengths = _record_len_to_list(record_len)
        if not lengths:
            lengths = [features.shape[0]]
        split_features = self._split(features, lengths)
        split_demand_score = self._split(demand_score, lengths)
        split_demand_mask = self._split(demand_mask.float(), lengths)
        split_supply = self._split(teacher_supply, lengths)
        split_supply_mask = self._split(teacher_supply_mask.float(), lengths)

        teacher_masks = []
        offset = 0
        for batch_idx, cav_num in enumerate(lengths):
            sample_actions = actions[offset: offset + cav_num]
            _, teacher_mask, _ = SupplyDemandLAMMAComm._mask_one_sample(
                self,
                split_features[batch_idx],
                split_demand_score[batch_idx],
                split_demand_mask[batch_idx],
                split_supply[batch_idx],
                split_supply_mask[batch_idx],
                affine_matrix[batch_idx, :cav_num, :cav_num],
                sample_actions,
            )
            teacher_masks.append(teacher_mask)
            offset += cav_num

        teacher_mask = torch.cat(teacher_masks, dim=0) > 0.25
        student_mask = broadcast_mask > 0.25
        overlap_cells = int((teacher_mask & student_mask).sum().item())
        teacher_cells = int(teacher_mask.sum().item())
        student_cells = int(student_mask.sum().item())
        return {
            "pairwise_teacher_overlap": float(overlap_cells / max(teacher_cells, 1)),
            "pairwise_teacher_selected_cells": teacher_cells,
            "broadcast_teacher_overlap_cells": overlap_cells,
            "broadcast_teacher_student_cells": student_cells,
        }

    def _should_save_debug_tensor(self, name: str) -> bool:
        if bool(self.debug_cfg.get("save_masks", False)):
            return True
        if name == "demand":
            return bool(self.broadcast_debug_cfg.get("save_broadcast_demand", False))
        if name == "mask":
            return bool(self.broadcast_debug_cfg.get("save_broadcast_mask", False))
        if name == "utility":
            return bool(self.broadcast_debug_cfg.get("save_broadcast_utility", False))
        if name == "receiver_gating":
            return bool(self.broadcast_debug_cfg.get("save_receiver_gating", False))
        return False

    def _maybe_log(self, debug: Dict[str, Any]):
        if not bool(self.debug_cfg.get("log", False)):
            return
        max_batches = self.debug_cfg.get("max_batches", 5)
        if max_batches is not None and self.debug_counter >= int(max_batches):
            self.debug_counter += 1
            return
        print(
            "[BROAD-SD-LAMMA] demand={:.4f} supply={:.4f} utility={:.4f} "
            "selected={:.4f} budget={:.4f} sender_packets={} payload_kbits={:.2f} gating={}".format(
                debug.get("broadcast_demand_mean", 0.0),
                debug.get("broadcast_supply_mean", 0.0),
                debug.get("broadcast_utility_mean", 0.0),
                debug.get("broadcast_selected_ratio", 0.0),
                debug.get("broadcast_budget_ratio", 0.0),
                debug.get("sender_packet_count", 0),
                debug.get("estimated_broadcast_payload_kbits", 0.0),
                debug.get("receiver_gating_enabled", False),
            )
        )
        self.debug_counter += 1
