import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .config import LightSADConfig
from .local_reliability import build_local_reliability
from .runtime_mask import expand_actions
from .sensor_stats import collect_light_sad_state, record_len_to_list, total_cavs


class LightSADDispatcher:
    """
    Light-SAD v1: per-CAV runtime modality scheduling prototype.

    This is not a full EMC2 implementation and does not import OpenPCDet code.
    This is not a full MoME implementation and does not add DETR/AQR/MED decoders.
    """

    def __init__(self, cfg=None):
        self.cfg = LightSADConfig.from_dict(cfg or {})
        self._warned = set()

    def dispatch(self, data_dict: dict, record_len=None) -> dict:
        state = collect_light_sad_state(data_dict, record_len=record_len, per_cav=self.cfg.per_cav)
        if self.cfg.use_local_reliability:
            local = build_local_reliability(
                data_dict,
                record_len=record_len,
                camera_stats_per_cav=state.get("per_cav", {}).get("camera", []),
            )
            state["per_cav"]["local_reliability"] = local

        total = total_cavs(record_len)
        global_state = self._global_state(state)
        global_action, global_reason = self.decide_action(global_state)
        result = {
            "action": global_action,
            "reason": global_reason,
            "state": state,
            "mode": "batch",
            "reliability": self._estimate_reliability(global_action, global_state),
        }

        if self.cfg.force_actions:
            actions = expand_actions(self.cfg.force_actions, total)
            cav_states = self._per_cav_states(state, total)
            result.update({
                "actions": actions,
                "reasons": ["force_actions_%s" % act for act in actions],
                "states": cav_states,
                "reliabilities": [
                    self._estimate_reliability(action, cav_state)
                    for action, cav_state in zip(actions, cav_states)
                ],
                "mode": "per_cav",
                "action": self._aggregate_actions(actions),
                "reason": "force_actions",
            })
        elif self.cfg.per_cav:
            if state.get("per_cav_valid", False):
                cav_states = self._per_cav_states(state, total)
                actions, reasons = [], []
                for cav_state in cav_states:
                    action, reason = self.decide_action(cav_state)
                    actions.append(action)
                    reasons.append(reason)
                result.update({
                    "actions": actions,
                    "reasons": reasons,
                    "states": cav_states,
                    "reliabilities": [
                        self._estimate_reliability(action, cav_state)
                        for action, cav_state in zip(actions, cav_states)
                    ],
                    "mode": "per_cav",
                    "action": self._aggregate_actions(actions),
                    "reason": "per_cav_%s" % self._aggregate_actions(actions),
                })
            else:
                result["fallback_reason"] = state.get("fallback_reasons", {})
                if self.cfg.log:
                    print("[Light-SAD] per-CAV fallback:", result["fallback_reason"])

        result["state_summary"] = self._state_summary(result)
        self._maybe_dump(result)
        return result

    def decide_action(self, state: dict) -> Tuple[str, str]:
        cfg = self.cfg
        if cfg.force_action in {"L", "C", "LC"} and not cfg.force_actions:
            return cfg.force_action, "force_action_%s" % cfg.force_action

        lidar = state.get("lidar", {})
        camera = state.get("camera", {})
        history = state.get("history", {}) if cfg.use_history else {}
        local = state.get("local_reliability", {})

        lidar_valid = bool(lidar.get("valid", False))
        camera_valid = bool(camera.get("valid", False))
        lidar_points = int(lidar.get("num_points", 0))
        lidar_voxels = int(lidar.get("num_voxels", 0))
        mean_points = float(lidar.get("mean_points_per_voxel", 0.0))
        distant_ratio = float(lidar.get("distant_voxel_ratio", 0.0))
        distant_sparse = float(lidar.get("distant_sparse_score", 0.0))
        dark_score = float(camera.get("dark_score", 0.0))
        blur_proxy = float(camera.get("blur_proxy", 0.0))
        contrast = float(camera.get("contrast", 0.0))

        lidar_good = lidar_valid and lidar_points >= cfg.lidar_good_points and lidar_voxels >= cfg.lidar_min_voxels
        lidar_weak = (not lidar_valid) or lidar_points < cfg.lidar_min_points or lidar_voxels < cfg.lidar_min_voxels
        camera_degraded = (not camera_valid) or dark_score >= 0.6 or blur_proxy < cfg.camera_blur_thr
        camera_good = camera_valid and not camera_degraded and (contrast + blur_proxy) >= 0.08
        distant_uncertain = distant_ratio >= cfg.distant_voxel_ratio_thr or distant_sparse >= cfg.distant_sparse_thr

        history_valid = bool(history.get("valid", False))
        history_score = float(history.get("last_topk_mean_score", history.get("last_mean_score", 0.0)))
        history_low = history_valid and history_score < cfg.history_low_conf_thr
        history_high = history_valid and history_score >= cfg.history_high_conf_thr

        local_summary = local.get("summary", local) if isinstance(local, dict) else {}
        local_lidar_low = float(local_summary.get("low_lidar_region_ratio", 0.0)) >= 0.45
        local_camera_ok = bool(local_summary.get("camera_reliable_flag", False))

        distance_proxy = max(distant_ratio, distant_sparse)
        density_proxy = min(mean_points / 5.0, 1.0) if lidar_valid else 0.0
        camera_proxy = 0.0 if camera_degraded else min(contrast + blur_proxy, 1.0)
        history_proxy = history_score if history_valid else 0.5
        clarity_proxy = 0.45 * density_proxy + 0.35 * camera_proxy + 0.20 * history_proxy

        state["distance_proxy"] = float(distance_proxy)
        state["clarity_proxy"] = float(clarity_proxy)

        if not lidar_valid and camera_valid:
            return "C", "lidar_invalid_camera_available"
        if camera_degraded and lidar_valid:
            return "L", "camera_degraded_lidar_preferred"
        if cfg.conservative_on_low_history and history_low and camera_valid:
            return "LC", "low_history_conservative_lc"
        if local_lidar_low and local_camera_ok:
            return "LC", "local_lidar_low_camera_reliable"
        if lidar_good and not distant_uncertain and (history_high or not history_valid):
            return "L", "close_distinct_good_lidar"
        if lidar_good and not distant_uncertain and clarity_proxy >= 0.55:
            return "L", "close_distinct_high_clarity"
        if lidar_weak and camera_good:
            return "LC", "distant_uncertain_or_weak_lidar_camera_good"
        if distant_uncertain and camera_good:
            return "LC", "mixed_visibility_distant_sparse"
        if lidar_valid and not camera_valid:
            return "L", "camera_invalid_lidar_available"
        if lidar_good:
            return "L", "good_lidar_low_cost"
        return "LC", "default_multimodal"

    def _global_state(self, state: dict) -> dict:
        return {
            "lidar": state.get("lidar", {}),
            "camera": state.get("camera", {}),
            "network": state.get("network", {}),
            "history": state.get("history", {}),
        }

    def _per_cav_states(self, state: dict, total: int) -> List[dict]:
        pc = state.get("per_cav", {})
        local = pc.get("local_reliability", [])
        history = state.get("history", {})
        states = []
        for idx in range(total):
            cav_state = {
                "lidar": self._safe_get(pc.get("lidar", []), idx, state.get("lidar", {})),
                "camera": self._safe_get(pc.get("camera", []), idx, state.get("camera", {})),
                "network": self._safe_get(pc.get("network", []), idx, state.get("network", {})),
                "history": history,
            }
            if idx < len(local):
                cav_state["local_reliability"] = local[idx]
            states.append(cav_state)
        return states

    @staticmethod
    def _safe_get(values, idx, default):
        return values[idx] if isinstance(values, list) and idx < len(values) else default

    @staticmethod
    def _aggregate_actions(actions: List[str]) -> str:
        if not actions:
            return "LC"
        if all(action == actions[0] for action in actions):
            return actions[0]
        return "LC"

    def _estimate_reliability(self, action: str, state: dict) -> float:
        lidar = state.get("lidar", {}) if isinstance(state, dict) else {}
        camera = state.get("camera", {}) if isinstance(state, dict) else {}
        history = state.get("history", {}) if isinstance(state, dict) else {}
        local = state.get("local_reliability", {}) if isinstance(state, dict) else {}

        lidar_rel = self._lidar_reliability(lidar, local)
        camera_rel = self._camera_reliability(camera, local)
        history_rel = self._history_reliability(history)

        action = str(action or "LC").upper()
        if action == "L":
            base = lidar_rel
        elif action == "C":
            base = camera_rel
        else:
            base = 0.45 * lidar_rel + 0.45 * camera_rel + 0.10 * max(lidar_rel, camera_rel)

        return float(max(0.0, min(1.0, 0.85 * base + 0.15 * history_rel)))

    def _lidar_reliability(self, lidar: dict, local: dict) -> float:
        if not bool(lidar.get("valid", False)):
            return 0.35
        points = float(lidar.get("num_points", 0.0))
        voxels = float(lidar.get("num_voxels", 0.0))
        mean_points = float(lidar.get("mean_points_per_voxel", 0.0))
        sparse = float(lidar.get("distant_sparse_score", 0.0))
        point_score = min(points / max(float(self.cfg.lidar_good_points), 1.0), 1.0)
        voxel_score = min(voxels / max(float(self.cfg.lidar_min_voxels), 1.0), 1.0)
        density_score = min(mean_points / 5.0, 1.0)
        rel = 0.35 * point_score + 0.30 * voxel_score + 0.25 * density_score + 0.10 * (1.0 - min(sparse, 1.0))
        local_summary = local.get("summary", local) if isinstance(local, dict) else {}
        if local_summary:
            rel *= 1.0 - 0.35 * min(float(local_summary.get("low_lidar_region_ratio", 0.0)), 1.0)
        return float(max(0.0, min(1.0, rel)))

    def _camera_reliability(self, camera: dict, local: dict) -> float:
        if not bool(camera.get("valid", False)):
            return 0.35
        dark = min(float(camera.get("dark_score", 0.0)), 1.0)
        blur = min(float(camera.get("blur_proxy", 0.0)) / max(float(self.cfg.camera_blur_thr), 1.0e-6), 1.0)
        contrast = min(float(camera.get("contrast", 0.0)) / max(float(self.cfg.camera_good_brightness_thr), 1.0e-6), 1.0)
        rel = 0.35 * (1.0 - dark) + 0.35 * blur + 0.30 * contrast
        local_summary = local.get("summary", local) if isinstance(local, dict) else {}
        if local_summary and bool(local_summary.get("camera_reliable_flag", False)):
            rel = max(rel, 0.75)
        return float(max(0.0, min(1.0, rel)))

    @staticmethod
    def _history_reliability(history: dict) -> float:
        if not isinstance(history, dict) or not bool(history.get("valid", False)):
            return 1.0
        score = float(history.get("last_topk_mean_score", history.get("last_mean_score", 0.5)))
        return float(max(0.0, min(1.0, score)))

    def _state_summary(self, result: dict) -> dict:
        actions = result.get("actions", [result.get("action", "LC")])
        summary = {
            "mode": result.get("mode", "batch"),
            "actions": actions,
            "fallback_reason": result.get("fallback_reason", None),
        }
        states = result.get("states", [])
        if states:
            summary["num_cavs"] = len(states)
            summary["mean_distant_sparse_score"] = sum(float(s.get("lidar", {}).get("distant_sparse_score", 0.0)) for s in states) / len(states)
            summary["mean_camera_dark_score"] = sum(float(s.get("camera", {}).get("dark_score", 0.0)) for s in states) / len(states)
            local_values = []
            for s in states:
                local_summary = s.get("local_reliability", {}).get("summary", {})
                if local_summary:
                    local_values.append(float(local_summary.get("low_lidar_region_ratio", 0.0)))
            if local_values:
                summary["mean_low_lidar_region_ratio"] = sum(local_values) / len(local_values)
        return summary

    def _maybe_dump(self, result: dict):
        if not self.cfg.debug_dump_state:
            return
        path = Path(self.cfg.debug_dump_path or "light_sad_state.jsonl")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(self._jsonable(result), sort_keys=True) + "\n")
        except Exception as exc:
            key = "dump_failed"
            if key not in self._warned:
                self._warned.add(key)
                print("[Light-SAD] debug dump failed:", exc)

    def _jsonable(self, obj):
        if torch.is_tensor(obj):
            return {
                "tensor_shape": list(obj.shape),
                "tensor_mean": float(obj.float().mean().item()) if obj.numel() > 0 else 0.0,
            }
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._jsonable(v) for v in obj]
        return obj
