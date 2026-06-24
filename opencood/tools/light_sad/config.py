from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class LightSADConfig:
    enabled: bool = False
    policy: str = "emc2_rule"
    force_action: Optional[str] = None
    log: bool = False

    # LiDAR quality thresholds
    lidar_min_points: int = 1500
    lidar_good_points: int = 8000
    lidar_min_voxels: int = 300

    # Camera quality thresholds
    camera_dark_thr: float = 0.12
    camera_blur_thr: float = 0.02
    camera_good_brightness_thr: float = 0.20

    # Network/compute thresholds
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
        return cls(**values)
