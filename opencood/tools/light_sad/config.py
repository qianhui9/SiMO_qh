from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Optional
import warnings


@dataclass
class LightSADConfig:
    enabled: bool = False
    policy: str = "emc2_rule"
    force_action: Optional[str] = None
    force_actions: Optional[str] = None
    log: bool = False

    # Light-SAD v1 controls.
    per_cav: bool = True
    use_history: bool = False
    use_local_reliability: bool = False
    history_low_conf_thr: float = 0.35
    history_high_conf_thr: float = 0.65
    history_topk: int = 20
    history_stale_limit: int = 3
    distant_voxel_ratio_thr: float = 0.35
    distant_sparse_thr: float = 0.22
    mixed_visibility_margin: float = 0.15
    conservative_on_low_history: bool = True
    debug_dump_state: bool = False
    debug_dump_path: Optional[str] = None

    # LiDAR quality thresholds.
    lidar_min_points: int = 1500
    lidar_good_points: int = 8000
    lidar_min_voxels: int = 300

    # Camera quality thresholds.
    camera_dark_thr: float = 0.12
    camera_blur_thr: float = 0.02
    camera_good_brightness_thr: float = 0.20

    # Network/compute placeholders. This prototype keeps real network traces out.
    low_bandwidth_mbps: float = 5.0
    high_rtt_ms: float = 100.0
    deadline_ms: float = 100.0

    @classmethod
    def from_dict(cls, cfg: dict):
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg

        field_names = {field.name for field in fields(cls)}
        values = {}
        if isinstance(cfg, Mapping) or hasattr(cfg, "get"):
            missing = object()
            for name in field_names:
                value = cfg.get(name, missing)
                if value is not missing:
                    values[name] = value

        obj = cls(**values)
        obj.normalize()
        return obj

    def normalize(self):
        valid_policies = {
            "force",
            "emc2_rule",
            "emc2_rule_history",
            "emc2_rule_local",
            "emc2_rule_full",
        }
        if self.policy not in valid_policies:
            warnings.warn(
                "Unknown Light-SAD policy '%s', falling back to emc2_rule." % self.policy
            )
            self.policy = "emc2_rule"

        if self.policy == "force" and not (self.force_action or self.force_actions):
            warnings.warn("Light-SAD policy=force but no force action was provided.")
        if self.policy in {"emc2_rule_history", "emc2_rule_full"} and not self.use_history:
            warnings.warn("Light-SAD history policy requested with use_history=False; history is ignored.")
        if self.policy in {"emc2_rule_local", "emc2_rule_full"} and not self.use_local_reliability:
            warnings.warn("Light-SAD local policy requested with use_local_reliability=False; local reliability is ignored.")
        return self
