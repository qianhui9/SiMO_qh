# -*- coding: utf-8 -*-
"""Supply-demand-aware LAMMA communication masks.

This module runs after SiMO/LAMMA has produced aligned BEV features and before
collaborative Pyramid Fusion. The first implementation keeps the dense tensor
interface unchanged and estimates sparse communication cost from the selected
mask coverage.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def _record_len_to_list(record_len) -> List[int]:
    if record_len is None:
        return []
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def _nested_get(cfg: Dict[str, Any], *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        value = value.detach().float().mean().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class SupplyDemandLAMMAComm(nn.Module):
    """
    Receiver-conditioned SD-LAMMA communication scheduler.

    Input features are flattened CAV features with shape [sum(record_len), C, H, W].
    For each batch item, ego is index 0, matching PyramidFusion.forward_collab.
    Collaborator masks are selected in the ego BEV frame and warped back to each
    sender frame before dense zero masking.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))

        self.demand_cfg = self.cfg.get("demand", {})
        self.supply_cfg = self.cfg.get("supply", {})
        self.network_cfg = self.cfg.get("network", {})
        self.redundancy_cfg = self.cfg.get("redundancy", {})
        self.mask_cfg = self.cfg.get("mask", {})
        self.debug_cfg = self.cfg.get("debug", {})

        self.align_corners = bool(self.mask_cfg.get("align_corners", False))
        self.debug_counter = 0
        self._active_network_state = {}
        self._active_channels = 1
        self._active_dtype_bits = 32

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
            return features, {"enabled": False}

        if features.dim() != 4:
            raise ValueError("SD-LAMMA expects [sum_cav, C, H, W] features.")

        lengths = _record_len_to_list(record_len)
        if not lengths:
            lengths = [features.shape[0]]
        total_cavs = int(sum(lengths))
        if total_cavs != features.shape[0]:
            raise ValueError(
                "SD-LAMMA record_len sum %d does not match features %d."
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
        lidar_active = self._extract_lidar_active(runtime_modality_mask, total_cavs, features.device)

        demand_score = self._build_demand_score(
            features, confidence, record_len, data_dict, lidar_active, light_sad_info
        )
        demand_mask = self._binary_from_score(
            demand_score,
            threshold=float(self.demand_cfg.get("uncertainty_threshold", 0.5)),
            topk_ratio=self.demand_cfg.get("topk_ratio", None),
        )

        supply_score = self._build_supply_score(confidence, reliability)
        supply_mask = self._binary_from_score(
            supply_score,
            threshold=float(self.supply_cfg.get("confidence_threshold", 0.01)),
            topk_ratio=self.supply_cfg.get("topk_ratio", None),
        )

        split_features = self._split(features, lengths)
        split_demand_score = self._split(demand_score, lengths)
        split_demand_mask = self._split(demand_mask.float(), lengths)
        split_supply_score = self._split(supply_score, lengths)
        split_supply_mask = self._split(supply_mask.float(), lengths)

        out_features = []
        out_masks = []
        sample_summaries = []
        offset = 0
        for batch_idx, cav_num in enumerate(lengths):
            sample_actions = actions[offset: offset + cav_num]
            sample_features, sample_mask, sample_debug = self._mask_one_sample(
                split_features[batch_idx],
                split_demand_score[batch_idx],
                split_demand_mask[batch_idx],
                split_supply_score[batch_idx],
                split_supply_mask[batch_idx],
                affine_matrix[batch_idx, :cav_num, :cav_num],
                sample_actions,
            )
            out_features.append(sample_features)
            out_masks.append(sample_mask)
            sample_summaries.append(sample_debug)
            offset += cav_num

        masked_features = torch.cat(out_features, dim=0)
        comm_mask = torch.cat(out_masks, dim=0)
        debug = self._summarize(
            sample_summaries,
            comm_mask,
            features.shape[1],
            features.dtype,
            actions,
        )
        debug["modality_reliability"] = [
            float(x) for x in reliability.detach().view(-1).cpu().tolist()
        ]
        if bool(self.debug_cfg.get("save_masks", False)):
            debug["communication_mask"] = comm_mask.detach().cpu()
            debug["demand_score"] = demand_score.detach().cpu()
            debug["supply_score"] = supply_score.detach().cpu()
            if bool(self.mask_cfg.get("multiscale", True)):
                debug["multiscale_communication_mask"] = self._make_multiscale_masks(comm_mask)

        if bool(self.mask_cfg.get("export_sparse", False)):
            debug.update(self._dense_to_sparse_debug(masked_features, comm_mask))
        self._maybe_log(debug)
        return masked_features, debug

    def export_pairwise_teacher(
        self,
        features: torch.Tensor,
        record_len,
        affine_matrix: torch.Tensor,
        data_dict: Optional[Dict[str, Any]] = None,
        light_sad_info: Optional[Dict[str, Any]] = None,
        runtime_modality_mask: Optional[Dict[str, torch.Tensor]] = None,
        confidence: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        actions: Optional[List[str]] = None,
        confidence_head: Optional[nn.Module] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export no-grad pairwise teacher masks in sender coordinates."""
        if not self.enabled or features is None:
            return None
        if features.dim() != 4:
            raise ValueError("SD-LAMMA teacher export expects [sum_cav, C, H, W] features.")

        lengths = _record_len_to_list(record_len)
        if not lengths:
            lengths = [features.shape[0]]
        total_cavs = int(sum(lengths))
        if total_cavs != features.shape[0]:
            raise ValueError(
                "SD-LAMMA record_len sum %d does not match features %d."
                % (total_cavs, features.shape[0])
            )

        self._active_network_state = self._extract_network_state(data_dict, light_sad_info)
        self._active_channels = int(features.shape[1])
        self._active_dtype_bits = self._dtype_bits(features.dtype)

        if confidence is None:
            confidence = self._build_confidence(features, confidence_head)
        if reliability is None:
            reliability = self._extract_reliability(
                light_sad_info,
                runtime_modality_mask,
                total_cavs,
                features.device,
                features.dtype,
            )
        if actions is None:
            actions = self._extract_actions(light_sad_info, total_cavs)

        lidar_active = self._extract_lidar_active(runtime_modality_mask, total_cavs, features.device)
        demand_score = self._build_demand_score(
            features, confidence, record_len, data_dict, lidar_active, light_sad_info
        )
        demand_mask = self._binary_from_score(
            demand_score,
            threshold=float(self.demand_cfg.get("uncertainty_threshold", 0.5)),
            topk_ratio=self.demand_cfg.get("topk_ratio", None),
        )
        supply_score = self._build_supply_score(confidence, reliability)
        supply_mask = self._binary_from_score(
            supply_score,
            threshold=float(self.supply_cfg.get("confidence_threshold", 0.01)),
            topk_ratio=self.supply_cfg.get("topk_ratio", None),
        )

        split_features = self._split(features, lengths)
        split_demand_score = self._split(demand_score, lengths)
        split_demand_mask = self._split(demand_mask.float(), lengths)
        split_supply = self._split(supply_score, lengths)
        split_supply_mask = self._split(supply_mask.float(), lengths)

        teacher_masks = []
        teacher_utilities = []
        sample_summaries = []
        offset = 0
        for batch_idx, cav_num in enumerate(lengths):
            sample_actions = actions[offset: offset + cav_num]
            _, teacher_mask, sample_debug = self._mask_one_sample(
                split_features[batch_idx],
                split_demand_score[batch_idx],
                split_demand_mask[batch_idx],
                split_supply[batch_idx],
                split_supply_mask[batch_idx],
                affine_matrix[batch_idx, :cav_num, :cav_num],
                sample_actions,
            )
            teacher_masks.append(teacher_mask)
            teacher_utilities.append(
                self._pairwise_teacher_utility_one_sample(
                    split_demand_score[batch_idx],
                    split_supply[batch_idx],
                    affine_matrix[batch_idx, :cav_num, :cav_num],
                )
            )
            sample_summaries.append(sample_debug)
            offset += cav_num

        teacher_mask = torch.cat(teacher_masks, dim=0).detach()
        teacher_utility = torch.cat(teacher_utilities, dim=0).detach().clamp(0.0, 1.0)
        debug = self._summarize(
            sample_summaries,
            teacher_mask,
            features.shape[1],
            features.dtype,
            actions,
        )
        return {
            "teacher_mask": teacher_mask,
            "teacher_utility": teacher_utility,
            "teacher_selected_ratio": debug.get("communication_rate", 0.0),
            "teacher_selected_cells": debug.get("selected_cells", 0),
            "teacher_total_collab_cells": debug.get("total_collab_cells", 0),
            "teacher_debug": debug,
        }

    def _pairwise_teacher_utility_one_sample(
        self,
        demand_score: torch.Tensor,
        supply_score: torch.Tensor,
        t_matrix: torch.Tensor,
    ) -> torch.Tensor:
        cav_num, _, height, width = supply_score.shape
        utility_sender = torch.zeros_like(supply_score)
        if cav_num <= 1:
            return utility_sender
        supply_in_ego = warp_affine_simple(
            supply_score,
            t_matrix[0, :cav_num],
            (height, width),
            align_corners=self.align_corners,
        )
        utility_ego = supply_in_ego * demand_score[0:1]
        utility_ego[0].zero_()
        utility_sender = warp_affine_simple(
            utility_ego,
            t_matrix[:cav_num, 0],
            (height, width),
            align_corners=self.align_corners,
        ).clamp(0.0, 1.0)
        utility_sender[0].zero_()
        return utility_sender

    @staticmethod
    def _split(x: torch.Tensor, lengths: List[int]) -> List[torch.Tensor]:
        if len(lengths) == 1:
            return [x]
        splits = torch.tensor(lengths[:-1], device=x.device).cumsum(dim=0).cpu()
        return list(torch.tensor_split(x, splits))

    def _build_confidence(
        self,
        features: torch.Tensor,
        confidence_head: Optional[nn.Module],
    ) -> torch.Tensor:
        confidence = None
        if confidence_head is not None:
            try:
                with torch.no_grad():
                    logits = confidence_head(features)
                    confidence = torch.sigmoid(logits.detach())
                    if confidence.dim() == 4 and confidence.shape[1] > 1:
                        confidence = confidence.max(dim=1, keepdim=True).values
            except Exception:
                confidence = None

        if confidence is None or confidence.dim() != 4:
            energy = features.detach().float().abs().mean(dim=1, keepdim=True)
            confidence = self._normalize_map(energy)

        if confidence.shape[-2:] != features.shape[-2:]:
            confidence = F.interpolate(
                confidence,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return confidence.clamp(0.0, 1.0).to(device=features.device, dtype=features.dtype)

    def _build_demand_score(
        self,
        features: torch.Tensor,
        confidence: torch.Tensor,
        record_len,
        data_dict: Optional[Dict[str, Any]],
        lidar_active: torch.Tensor,
        light_sad_info: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        score = torch.zeros_like(confidence)
        weight = torch.zeros((features.shape[0], 1, 1, 1), device=features.device, dtype=features.dtype)
        uncertainty = 1.0 - confidence
        density_demand_for_occ = None

        if bool(self.demand_cfg.get("use_uncertainty", True)):
            w = float(self.demand_cfg.get("uncertainty_weight", 0.55))
            score = score + uncertainty * w
            weight = weight + w

        if bool(self.demand_cfg.get("use_lidar_density", True)) or bool(self.demand_cfg.get("use_occlusion", False)):
            density = self._build_lidar_density_maps(data_dict, record_len, features)
            if density is not None:
                density_threshold = float(self.demand_cfg.get("density_threshold", 0.125))
                if density_threshold > 0:
                    density_demand = torch.clamp((density_threshold - density) / density_threshold, 0.0, 1.0)
                else:
                    density_demand = 1.0 - density
                active = lidar_active.to(device=features.device, dtype=features.dtype).view(-1, 1, 1, 1)
                density_demand_for_occ = density_demand * active
                if bool(self.demand_cfg.get("use_lidar_density", True)):
                    w = float(self.demand_cfg.get("density_weight", 0.35))
                    score = score + density_demand_for_occ * w
                    weight = weight + active * w

        if bool(self.demand_cfg.get("use_occlusion", False)):
            # First-stage occlusion proxy: regions that are simultaneously sparse
            # in ego LiDAR and uncertain in the single-agent BEV head. Camera-only
            # agents fall back to uncertainty so demand never collapses to zero.
            if density_demand_for_occ is None:
                occlusion = uncertainty
            else:
                occlusion = torch.sqrt(torch.clamp(density_demand_for_occ * uncertainty, 0.0, 1.0))
            w = float(self.demand_cfg.get("occlusion_weight", 0.15))
            score = score + occlusion * w
            weight = weight + w

        if bool(self.demand_cfg.get("use_history", False)):
            history = self._history_demand(light_sad_info, features.shape[0], features.device, features.dtype)
            if history is not None:
                w = float(self.demand_cfg.get("history_weight", 0.10))
                score = score + history * w
                weight = weight + w

        fallback = 1.0 - confidence
        return torch.where(weight > 1e-6, score / torch.clamp(weight, min=1e-6), fallback).clamp(0.0, 1.0)

    def _build_lidar_density_maps(
        self,
        data_dict: Optional[Dict[str, Any]],
        record_len,
        features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if data_dict is None:
            return None
        processed = data_dict.get("processed_lidar", None)
        if not isinstance(processed, dict):
            processed = data_dict.get("inputs_m1", None)
        if not isinstance(processed, dict):
            return None

        coords = processed.get("voxel_coords", None)
        if coords is None or not torch.is_tensor(coords) or coords.numel() == 0 or coords.dim() < 2:
            return None

        coords = coords.to(features.device)
        total, _, height, width = features.shape
        lengths = _record_len_to_list(record_len)
        if not lengths:
            lengths = [total]

        density = torch.zeros((total, 1, height, width), device=features.device, dtype=features.dtype)
        point_counts = processed.get("voxel_num_points", None)
        if torch.is_tensor(point_counts) and point_counts.shape[0] == coords.shape[0]:
            values = point_counts.to(device=features.device, dtype=features.dtype).view(-1)
        else:
            values = torch.ones((coords.shape[0],), device=features.device, dtype=features.dtype)

        first_col = coords[:, 0].long()
        max_index = int(first_col.max().item()) if first_col.numel() > 0 else -1
        if max_index < total:
            group_indices = [first_col == idx for idx in range(total)]
        elif max_index < len(lengths):
            group_indices = []
            for batch_idx, cav_num in enumerate(lengths):
                sample_mask = first_col == batch_idx
                group_indices.extend([sample_mask for _ in range(cav_num)])
        else:
            group_indices = [torch.ones_like(first_col, dtype=torch.bool) for _ in range(total)]

        grid_y, grid_x = self._lidar_grid_size(coords)
        y_idx = coords[:, -2].float()
        x_idx = coords[:, -1].float()
        fy = torch.clamp((y_idx / max(grid_y, 1.0) * height).long(), min=0, max=height - 1)
        fx = torch.clamp((x_idx / max(grid_x, 1.0) * width).long(), min=0, max=width - 1)

        for cav_idx, mask in enumerate(group_indices[:total]):
            if mask.sum() <= 0:
                continue
            density[cav_idx, 0].index_put_((fy[mask], fx[mask]), values[mask], accumulate=True)

        return self._normalize_map(density)

    def _lidar_grid_size(self, coords: torch.Tensor) -> Tuple[float, float]:
        lidar_range = self.cfg.get("lidar_range", None)
        voxel_size = self.cfg.get("voxel_size", None)
        if lidar_range is not None and voxel_size is not None:
            grid_x = (float(lidar_range[3]) - float(lidar_range[0])) / float(voxel_size[0])
            grid_y = (float(lidar_range[4]) - float(lidar_range[1])) / float(voxel_size[1])
            return max(grid_y, 1.0), max(grid_x, 1.0)
        grid_y = float(coords[:, -2].max().item() + 1) if coords.numel() > 0 else 1.0
        grid_x = float(coords[:, -1].max().item() + 1) if coords.numel() > 0 else 1.0
        return max(grid_y, 1.0), max(grid_x, 1.0)

    def _history_demand(
        self,
        light_sad_info: Optional[Dict[str, Any]],
        total: int,
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        if not isinstance(light_sad_info, dict):
            return None
        states = light_sad_info.get("states", None)
        if not isinstance(states, list) or not states:
            states = [light_sad_info.get("state", {}) for _ in range(total)]
        values = []
        for idx in range(total):
            state = states[idx] if idx < len(states) else {}
            history = state.get("history", {}) if isinstance(state, dict) else {}
            if not history or not history.get("valid", False):
                values.append(0.0)
                continue
            conf = _safe_float(history.get("last_topk_mean_score", history.get("last_mean_score", 0.0)), 0.0)
            values.append(max(0.0, min(1.0, 1.0 - conf)))
        return torch.tensor(values, device=device, dtype=dtype).view(total, 1, 1, 1)

    def _build_supply_score(self, confidence: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        score = confidence
        if bool(self.supply_cfg.get("use_modality_reliability", True)):
            score = score * reliability.to(device=confidence.device, dtype=confidence.dtype)
        return score.clamp(0.0, 1.0)

    def _mask_one_sample(
        self,
        features: torch.Tensor,
        demand_score: torch.Tensor,
        demand_mask: torch.Tensor,
        supply_score: torch.Tensor,
        supply_mask: torch.Tensor,
        t_matrix: torch.Tensor,
        actions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cav_num, channels, height, width = features.shape
        apply_mask = torch.ones((cav_num, 1, height, width), device=features.device, dtype=features.dtype)
        zero_comm_mask = torch.zeros_like(apply_mask)

        if cav_num <= 1:
            return features, zero_comm_mask, {
                "demand_ratio": float(demand_mask[0].mean().item()) if cav_num else 0.0,
                "supply_ratio": 0.0,
                "selected_ratio": 0.0,
                "redundancy_before_ratio": 0.0,
                "selected_cells": 0,
                "total_collab_cells": 0,
                "per_modality": {},
            }

        supply_in_ego = warp_affine_simple(
            supply_score,
            t_matrix[0, :cav_num],
            (height, width),
            align_corners=self.align_corners,
        )
        supply_mask_in_ego = warp_affine_simple(
            supply_mask,
            t_matrix[0, :cav_num],
            (height, width),
            align_corners=self.align_corners,
        ) > 0.25

        ego_demand_score = demand_score[0:1]
        ego_demand_mask = demand_mask[0:1] > 0.25
        candidate_score = supply_in_ego * ego_demand_score
        candidate_mask = supply_mask_in_ego & ego_demand_mask
        candidate_mask[0].zero_()
        candidate_score[0].zero_()

        before_ratio = float(candidate_mask[1:].float().mean().item()) if cav_num > 1 else 0.0
        selected_ego = self._select_candidates(candidate_score, candidate_mask)
        selected_sender = warp_affine_simple(
            selected_ego.float(),
            t_matrix[:cav_num, 0],
            (height, width),
            align_corners=self.align_corners,
        )
        selected_sender = (selected_sender > 0.25).to(dtype=features.dtype)
        selected_sender[0].fill_(1.0)

        masked_features = features * selected_sender
        comm_mask = zero_comm_mask
        comm_mask[1:] = selected_sender[1:]

        selected_cells = int(comm_mask[1:].sum().item())
        total_collab_cells = int((cav_num - 1) * height * width)
        per_modality = defaultdict(lambda: {"selected": 0.0, "total": 0.0})
        for cav_idx in range(1, cav_num):
            mode = actions[cav_idx] if cav_idx < len(actions) else "LC"
            per_modality[mode]["selected"] += float(comm_mask[cav_idx].sum().item())
            per_modality[mode]["total"] += float(height * width)

        return masked_features, comm_mask, {
            "demand_ratio": float(ego_demand_mask.float().mean().item()),
            "supply_ratio": float(supply_mask[1:].mean().item()),
            "selected_ratio": float(selected_cells / max(total_collab_cells, 1)),
            "redundancy_before_ratio": before_ratio,
            "selected_cells": selected_cells,
            "total_collab_cells": total_collab_cells,
            "per_modality": dict(per_modality),
        }

    def _select_candidates(self, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(self.redundancy_cfg.get("enabled", True)):
            return self._select_redundancy_aware(score, mask)
        return self._apply_global_budget(score, mask)

    def _select_redundancy_aware(self, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cav_num, _, height, width = score.shape
        selected = torch.zeros_like(mask, dtype=torch.bool)
        if cav_num <= 1:
            return selected

        collab_mask = mask[1:] & (score[1:] > 0.0)
        if not collab_mask.any():
            return selected

        candidate_space = max((cav_num - 1) * height * width, 1)
        budget = self._message_budget(candidate_space)
        if budget is not None and budget <= 0:
            return selected

        allow_overlap = bool(self.redundancy_cfg.get("allow_overlap", False))
        demand_decay = max(0.0, min(1.0, float(self.redundancy_cfg.get("demand_decay", 1.0))))
        flat_score = score[1:].reshape(-1)
        flat_mask = collab_mask.reshape(-1)
        candidate_idx = torch.nonzero(flat_mask, as_tuple=False).view(-1)
        if candidate_idx.numel() == 0:
            return selected

        order = torch.argsort(flat_score[candidate_idx], descending=True)
        ordered_idx = candidate_idx[order]
        remaining = torch.ones((height * width,), device=score.device, dtype=score.dtype)
        selected_flat = selected[1:].reshape(-1)
        cells_per_cav = height * width
        selected_count = 0

        for idx_tensor in ordered_idx:
            flat_idx = int(idx_tensor.item())
            cell_idx = flat_idx % cells_per_cav
            if remaining[cell_idx] <= 1e-6:
                continue
            gain = flat_score[flat_idx] * remaining[cell_idx]
            if gain <= 0:
                continue
            selected_flat[flat_idx] = True
            selected_count += 1

            if allow_overlap:
                remaining[cell_idx] = torch.clamp(remaining[cell_idx] - demand_decay, min=0.0)
            else:
                remaining[cell_idx] = 0.0

            if budget is not None and selected_count >= budget:
                break

        return selected

    def _apply_global_budget(self, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = mask & (score > 0.0)
        collab_cells = max((score.shape[0] - 1) * score.shape[-2] * score.shape[-1], 1)
        budget = self._message_budget(collab_cells)
        if budget is None:
            return selected
        return self._topk_mask(score, selected, budget)

    def _message_budget(self, candidate_cells: int) -> Optional[int]:
        ratio = self._effective_budget_ratio(candidate_cells)
        if ratio is None:
            return None
        return int(torch.ceil(torch.tensor(float(candidate_cells) * ratio)).item())

    def _effective_budget_ratio(self, candidate_cells: int) -> Optional[float]:
        ratios = []
        max_ratio = self.network_cfg.get("max_comm_ratio", None)
        budget_mode = str(self.network_cfg.get("budget_mode", "threshold"))
        if max_ratio is not None:
            ratios.append(max(0.0, min(1.0, float(max_ratio))))
        elif budget_mode == "topk":
            ratios.append(1.0)

        bandwidth_mbps = self._network_value("bandwidth_mbps")
        if bandwidth_mbps is not None:
            frame_rate_hz = max(float(self.network_cfg.get("frame_rate_hz", 10.0)), 1.0e-6)
            frame_budget_s = 1.0 / frame_rate_hz
            deadline_ms = self._network_value("deadline_ms")
            if deadline_ms is not None and deadline_ms > 0:
                frame_budget_s = min(frame_budget_s, float(deadline_ms) / 1000.0)
            latency_ms = self._network_value("latency_ms")
            if latency_ms is None:
                latency_ms = self._network_value("rtt_ms", default=0.0)
            tx_time_s = max(0.0, frame_budget_s - float(latency_ms or 0.0) / 1000.0)
            packet_loss = max(0.0, min(1.0, float(self._network_value("packet_loss", default=0.0) or 0.0)))
            payload_bits = max(float(bandwidth_mbps), 0.0) * 1.0e6 * tx_time_s * (1.0 - packet_loss)
            dense_bits = max(float(candidate_cells * self._active_channels * self._active_dtype_bits), 1.0)
            ratios.append(max(0.0, min(1.0, payload_bits / dense_bits)))

        if not ratios:
            return None
        return min(ratios)

    def _extract_network_state(
        self,
        data_dict: Optional[Dict[str, Any]],
        light_sad_info: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        state: Dict[str, float] = {}
        if isinstance(light_sad_info, dict):
            sad_state = light_sad_info.get("state", {})
            if isinstance(sad_state, dict) and isinstance(sad_state.get("network", None), dict):
                state.update(sad_state["network"])
        if isinstance(data_dict, dict) and isinstance(data_dict.get("network_state", None), dict):
            state.update(data_dict["network_state"])
        return state

    def _network_value(self, key: str, default=None):
        if key in self.network_cfg and self.network_cfg.get(key) is not None:
            return self.network_cfg.get(key)
        if key in self._active_network_state and self._active_network_state.get(key) is not None:
            return self._active_network_state.get(key)
        return default

    @staticmethod
    def _topk_mask(score: torch.Tensor, mask: torch.Tensor, budget: int) -> torch.Tensor:
        flat_mask = mask.reshape(-1).bool()
        out = torch.zeros_like(flat_mask)
        if budget <= 0 or not flat_mask.any():
            return out.view_as(mask)
        candidate_idx = torch.nonzero(flat_mask, as_tuple=False).view(-1)
        if candidate_idx.numel() <= budget:
            out[candidate_idx] = True
            return out.view_as(mask)
        candidate_score = score.reshape(-1)[candidate_idx]
        _, top_order = torch.topk(candidate_score, k=budget, largest=True, sorted=False)
        out[candidate_idx[top_order]] = True
        return out.view_as(mask)

    def _binary_from_score(self, score: torch.Tensor, threshold: float, topk_ratio=None) -> torch.Tensor:
        if topk_ratio is None:
            return (score >= threshold).to(dtype=score.dtype)

        ratio = max(0.0, min(1.0, float(topk_ratio)))
        cells = score.shape[-2] * score.shape[-1]
        budget = int(torch.ceil(torch.tensor(float(cells) * ratio)).item())
        masks = []
        for idx in range(score.shape[0]):
            valid = torch.ones_like(score[idx:idx + 1], dtype=torch.bool)
            masks.append(self._topk_mask(score[idx:idx + 1], valid, budget).to(dtype=score.dtype))
        return torch.cat(masks, dim=0)

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=2)
        min_v = flat.min(dim=-1).values.view(x.shape[0], x.shape[1], 1, 1)
        max_v = flat.max(dim=-1).values.view(x.shape[0], x.shape[1], 1, 1)
        return torch.where(max_v > min_v, (x - min_v) / torch.clamp(max_v - min_v, min=1e-6), torch.zeros_like(x))

    def _extract_actions(self, light_sad_info: Optional[Dict[str, Any]], total: int) -> List[str]:
        if isinstance(light_sad_info, dict):
            actions = light_sad_info.get("actions", None)
            if isinstance(actions, list) and actions:
                return [str(a) for a in actions[:total]] + ["LC"] * max(0, total - len(actions))
            action = light_sad_info.get("action", "LC")
            return [str(action) for _ in range(total)]
        return ["LC" for _ in range(total)]

    def _extract_reliability(
        self,
        light_sad_info: Optional[Dict[str, Any]],
        runtime_modality_mask: Optional[Dict[str, torch.Tensor]],
        total: int,
        device,
        dtype,
    ) -> torch.Tensor:
        min_reliability = float(self.supply_cfg.get("min_reliability", 0.0))
        values = None
        if isinstance(light_sad_info, dict):
            values = light_sad_info.get("reliabilities", light_sad_info.get("reliability", None))
        if values is None:
            rel = self._estimate_reliability_from_state(light_sad_info, total, device, dtype)
        else:
            rel = torch.as_tensor(values, device=device, dtype=dtype).view(-1)
            if rel.numel() < total:
                rel = torch.cat([rel, torch.ones((total - rel.numel(),), device=device, dtype=dtype)])
            rel = rel[:total]

        if runtime_modality_mask is not None and bool(self.supply_cfg.get("zero_inactive_modalities", False)):
            cam = runtime_modality_mask.get("camera", None)
            lidar = runtime_modality_mask.get("lidar", None)
            if cam is not None and lidar is not None:
                active = (cam.reshape(-1) + lidar.reshape(-1)).to(device=device, dtype=dtype)
                if active.numel() >= total:
                    rel = rel * (active[:total] > 0).to(dtype=dtype)
        rel = torch.clamp(rel, min=min_reliability, max=1.0)
        return rel.view(total, 1, 1, 1)

    def _estimate_reliability_from_state(self, light_sad_info, total: int, device, dtype) -> torch.Tensor:
        if not isinstance(light_sad_info, dict):
            return torch.ones((total,), device=device, dtype=dtype)

        states = light_sad_info.get("states", None)
        actions = self._extract_actions(light_sad_info, total)
        if not isinstance(states, list) or not states:
            base_state = light_sad_info.get("state", {})
            states = [base_state for _ in range(total)]

        values = []
        for idx in range(total):
            state = states[idx] if idx < len(states) and isinstance(states[idx], dict) else {}
            action = actions[idx] if idx < len(actions) else "LC"
            values.append(self._state_action_reliability(state, action))
        return torch.tensor(values, device=device, dtype=dtype)

    @staticmethod
    def _state_action_reliability(state: Dict[str, Any], action: str) -> float:
        lidar = state.get("lidar", {}) if isinstance(state, dict) else {}
        camera = state.get("camera", {}) if isinstance(state, dict) else {}
        history = state.get("history", {}) if isinstance(state, dict) else {}
        local = state.get("local_reliability", {}) if isinstance(state, dict) else {}
        lidar_rel = SupplyDemandLAMMAComm._state_lidar_reliability(lidar, local)
        camera_rel = SupplyDemandLAMMAComm._state_camera_reliability(camera, local)
        history_rel = 1.0
        if isinstance(history, dict) and bool(history.get("valid", False)):
            history_rel = max(0.0, min(1.0, _safe_float(history.get("last_topk_mean_score", history.get("last_mean_score", 0.5)), 0.5)))

        action = str(action or "LC").upper()
        if action == "L":
            base = lidar_rel
        elif action == "C":
            base = camera_rel
        else:
            base = 0.45 * lidar_rel + 0.45 * camera_rel + 0.10 * max(lidar_rel, camera_rel)
        return float(max(0.0, min(1.0, 0.85 * base + 0.15 * history_rel)))

    @staticmethod
    def _state_lidar_reliability(lidar: Dict[str, Any], local: Dict[str, Any]) -> float:
        if not bool(lidar.get("valid", False)):
            return 0.35
        points = _safe_float(lidar.get("num_points", 0.0), 0.0)
        voxels = _safe_float(lidar.get("num_voxels", 0.0), 0.0)
        mean_points = _safe_float(lidar.get("mean_points_per_voxel", 0.0), 0.0)
        sparse = min(_safe_float(lidar.get("distant_sparse_score", 0.0), 0.0), 1.0)
        point_score = min(points / 8000.0, 1.0)
        voxel_score = min(voxels / 300.0, 1.0)
        density_score = min(mean_points / 5.0, 1.0)
        rel = 0.35 * point_score + 0.30 * voxel_score + 0.25 * density_score + 0.10 * (1.0 - sparse)
        local_summary = local.get("summary", local) if isinstance(local, dict) else {}
        if local_summary:
            rel *= 1.0 - 0.35 * min(_safe_float(local_summary.get("low_lidar_region_ratio", 0.0), 0.0), 1.0)
        return float(max(0.0, min(1.0, rel)))

    @staticmethod
    def _state_camera_reliability(camera: Dict[str, Any], local: Dict[str, Any]) -> float:
        if not bool(camera.get("valid", False)):
            return 0.35
        dark = min(_safe_float(camera.get("dark_score", 0.0), 0.0), 1.0)
        blur = min(_safe_float(camera.get("blur_proxy", 0.0), 0.0) / 0.02, 1.0)
        contrast = min(_safe_float(camera.get("contrast", 0.0), 0.0) / 0.20, 1.0)
        rel = 0.35 * (1.0 - dark) + 0.35 * blur + 0.30 * contrast
        local_summary = local.get("summary", local) if isinstance(local, dict) else {}
        if local_summary and bool(local_summary.get("camera_reliable_flag", False)):
            rel = max(rel, 0.75)
        return float(max(0.0, min(1.0, rel)))

    @staticmethod
    def _extract_lidar_active(runtime_modality_mask, total: int, device) -> torch.Tensor:
        if isinstance(runtime_modality_mask, dict) and runtime_modality_mask.get("lidar", None) is not None:
            lidar = runtime_modality_mask["lidar"].reshape(-1).to(device=device)
            if lidar.numel() >= total:
                return (lidar[:total] > 0).float()
        return torch.ones((total,), device=device)

    @staticmethod
    def _dtype_bits(dtype) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            return 16
        if dtype in (torch.float64, torch.int64):
            return 64
        if dtype in (torch.int8, torch.uint8, torch.bool):
            return 8
        return 32

    def _make_multiscale_masks(self, mask: torch.Tensor) -> List[torch.Tensor]:
        masks = [mask.detach().cpu()]
        current = mask.detach().float()
        for _ in range(2):
            if current.shape[-1] <= 1 or current.shape[-2] <= 1:
                break
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            masks.append(current.cpu())
        return masks

    def _dense_to_sparse_debug(self, features: torch.Tensor, comm_mask: torch.Tensor) -> Dict[str, Any]:
        selected = comm_mask[:, 0] > 0
        indices = torch.nonzero(selected, as_tuple=False)
        values = features.permute(0, 2, 3, 1)[selected]
        output = {
            "sparse_indices": indices.detach().cpu(),
            "sparse_values": values.detach().cpu(),
            "sparse_shape": list(features.shape),
        }
        if bool(self.mask_cfg.get("return_dense_reconstructed_feature", False)):
            reconstructed = torch.zeros_like(features)
            reconstructed.permute(0, 2, 3, 1)[selected] = values
            output["dense_reconstructed_feature"] = reconstructed.detach().cpu()
        return output

    def _summarize(
        self,
        sample_summaries: List[Dict[str, Any]],
        comm_mask: torch.Tensor,
        channels: int,
        dtype,
        actions: List[str],
    ) -> Dict[str, Any]:
        selected_cells = sum(int(x.get("selected_cells", 0)) for x in sample_summaries)
        total_collab_cells = sum(int(x.get("total_collab_cells", 0)) for x in sample_summaries)
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

        def mean_value(key):
            vals = [float(x.get(key, 0.0)) for x in sample_summaries]
            return float(sum(vals) / len(vals)) if vals else 0.0

        return {
            "enabled": True,
            "num_samples": len(sample_summaries),
            "num_cavs": int(comm_mask.shape[0]),
            "mean_demand_ratio": mean_value("demand_ratio"),
            "mean_supply_ratio": mean_value("supply_ratio"),
            "mean_selected_ratio": mean_value("selected_ratio"),
            "mean_redundancy_before_ratio": mean_value("redundancy_before_ratio"),
            "communication_rate": float(selected_cells / max(total_collab_cells, 1)),
            "selected_cells": selected_cells,
            "total_collab_cells": total_collab_cells,
            "estimated_payload_bits": payload_bits,
            "estimated_payload_kbits": float(payload_bits / 1000.0),
            "per_modality_selected_ratio": per_modality_ratio,
            "actions": actions,
            "redundancy_enabled": bool(self.redundancy_cfg.get("enabled", True)),
            "dense_zero_mask": bool(self.mask_cfg.get("dense_zero_mask", True)),
        }

    def _maybe_log(self, debug: Dict[str, Any]):
        if not bool(self.debug_cfg.get("log", False)):
            return
        max_batches = self.debug_cfg.get("max_batches", 5)
        if max_batches is not None and self.debug_counter >= int(max_batches):
            self.debug_counter += 1
            return
        print(
            "[SD-LAMMA] demand={:.4f} supply={:.4f} selected={:.4f} "
            "before_redundancy={:.4f} payload_kbits={:.2f} modes={}".format(
                debug.get("mean_demand_ratio", 0.0),
                debug.get("mean_supply_ratio", 0.0),
                debug.get("mean_selected_ratio", 0.0),
                debug.get("mean_redundancy_before_ratio", 0.0),
                debug.get("estimated_payload_kbits", 0.0),
                debug.get("per_modality_selected_ratio", {}),
            )
        )
        self.debug_counter += 1
