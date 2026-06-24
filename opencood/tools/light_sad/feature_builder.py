import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


DEFAULT_FEATURE_NAMES = [
    "lidar.num_points",
    "lidar.num_voxels",
    "lidar.mean_points_per_voxel",
    "lidar.near_voxel_ratio",
    "lidar.distant_voxel_ratio",
    "lidar.distant_sparse_score",
    "lidar.valid",
    "lidar.reliability_score",
    "camera.brightness",
    "camera.contrast",
    "camera.blur_proxy",
    "camera.dark_score",
    "camera.overexposure_score",
    "camera.valid",
    "camera.reliability_score",
    "network.bandwidth_mbps",
    "network.rtt_ms",
    "network.packet_loss",
    "network.queue_delay_ms",
    "network.history_tx_delay_ms",
    "history.last_mean_score",
    "history.last_topk_mean_score",
    "history.confidence_trend",
    "history.valid",
    "history.last_num_detections",
    "local.low_lidar_region_ratio",
    "local.camera_reliable_flag",
    "local.collaborator_supply_score",
    "cav.index",
    "cav.is_ego",
    "cav.num_cavs",
    "cav.relative_distance_to_ego",
    "cav.ego_demand_mean",
    "rule.distance_proxy",
    "rule.clarity_proxy",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return float(default)
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.float().mean().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_float_list(value: Optional[Iterable[float]], length: int, fill: float) -> List[float]:
    if value is None:
        return [float(fill)] * length
    values = [_to_float(x, fill) for x in value]
    if len(values) != length:
        raise ValueError("Feature normalization length mismatch: expected %d, got %d" % (length, len(values)))
    return values


class LightSADFeatureBuilder:
    """
    Convert the lightweight Light-SAD state dictionary into a fixed-dimensional
    tensor. The builder only consumes cheap statistics collected before encoder
    execution.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_mean: Optional[Iterable[float]] = None,
        feature_std: Optional[Iterable[float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)
        self.dtype = dtype
        self.feature_mean = _as_float_list(feature_mean, len(self.feature_names), 0.0) if feature_mean is not None else None
        self.feature_std = _as_float_list(feature_std, len(self.feature_names), 1.0) if feature_std is not None else None

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def has_normalization(self) -> bool:
        return self.feature_mean is not None and self.feature_std is not None

    def set_normalization(self, mean: Iterable[float], std: Iterable[float]):
        self.feature_mean = _as_float_list(mean, self.dim, 0.0)
        self.feature_std = [max(x, 1.0e-6) for x in _as_float_list(std, self.dim, 1.0)]

    def feature_dict(self, state: Dict[str, Any]) -> Dict[str, float]:
        state = state or {}
        lidar = state.get("lidar", {}) or {}
        camera = state.get("camera", {}) or {}
        network = state.get("network", {}) or {}
        history = state.get("history", {}) or {}
        cav = state.get("cav", {}) or {}
        local = state.get("local_reliability", {}) or {}
        local_summary = local.get("summary", local) if isinstance(local, dict) else {}

        brightness = _to_float(camera.get("brightness", 0.0))
        contrast = _to_float(camera.get("contrast", 0.0))
        blur_proxy = _to_float(camera.get("blur_proxy", 0.0))
        dark_score = _to_float(camera.get("dark_score", 1.0))
        overexposure = _to_float(camera.get("overexposure_score", max(0.0, min(1.0, (brightness - 0.92) / 0.08))))
        camera_rel_default = max(0.0, min(1.0, 0.35 * (1.0 - dark_score) + 0.35 * min(blur_proxy / 0.02, 1.0) + 0.30 * min(contrast / 0.2, 1.0)))

        lidar_valid = bool(lidar.get("valid", False))
        lidar_points = _to_float(lidar.get("num_points", 0.0))
        lidar_voxels = _to_float(lidar.get("num_voxels", 0.0))
        mean_points = _to_float(lidar.get("mean_points_per_voxel", 0.0))
        distant_sparse = _to_float(lidar.get("distant_sparse_score", 0.0))
        lidar_rel_default = 0.0
        if lidar_valid:
            lidar_rel_default = max(0.0, min(1.0, 0.35 * min(lidar_points / 8000.0, 1.0) + 0.30 * min(lidar_voxels / 300.0, 1.0) + 0.25 * min(mean_points / 5.0, 1.0) + 0.10 * (1.0 - min(distant_sparse, 1.0))))

        last_mean = _to_float(history.get("last_mean_score", 0.0))
        last_topk = _to_float(history.get("last_topk_mean_score", last_mean))
        prev_mean = _to_float(history.get("prev_mean_score", last_mean))

        values = {
            "lidar.num_points": lidar_points,
            "lidar.num_voxels": lidar_voxels,
            "lidar.mean_points_per_voxel": mean_points,
            "lidar.near_voxel_ratio": _to_float(lidar.get("near_voxel_ratio", 0.0)),
            "lidar.distant_voxel_ratio": _to_float(lidar.get("distant_voxel_ratio", 0.0)),
            "lidar.distant_sparse_score": distant_sparse,
            "lidar.valid": 1.0 if lidar_valid else 0.0,
            "lidar.reliability_score": _to_float(lidar.get("reliability_score", lidar_rel_default)),
            "camera.brightness": brightness,
            "camera.contrast": contrast,
            "camera.blur_proxy": blur_proxy,
            "camera.dark_score": dark_score,
            "camera.overexposure_score": overexposure,
            "camera.valid": 1.0 if bool(camera.get("valid", False)) else 0.0,
            "camera.reliability_score": _to_float(camera.get("reliability_score", camera_rel_default)),
            "network.bandwidth_mbps": _to_float(network.get("bandwidth_mbps", 1000.0), 1000.0),
            "network.rtt_ms": _to_float(network.get("rtt_ms", network.get("latency_ms", 0.0))),
            "network.packet_loss": _to_float(network.get("packet_loss", 0.0)),
            "network.queue_delay_ms": _to_float(network.get("queue_delay_ms", 0.0)),
            "network.history_tx_delay_ms": _to_float(network.get("history_tx_delay_ms", 0.0)),
            "history.last_mean_score": last_mean,
            "history.last_topk_mean_score": last_topk,
            "history.confidence_trend": _to_float(history.get("confidence_trend", last_mean - prev_mean)),
            "history.valid": 1.0 if bool(history.get("valid", False)) else 0.0,
            "history.last_num_detections": _to_float(history.get("last_num_detections", 0.0)),
            "local.low_lidar_region_ratio": _to_float(local_summary.get("low_lidar_region_ratio", 0.0)),
            "local.camera_reliable_flag": 1.0 if bool(local_summary.get("camera_reliable_flag", False)) else 0.0,
            "local.collaborator_supply_score": _to_float(local_summary.get("collaborator_supply_score", 0.0)),
            "cav.index": _to_float(cav.get("index", 0.0)),
            "cav.is_ego": 1.0 if bool(cav.get("is_ego", False)) else 0.0,
            "cav.num_cavs": _to_float(cav.get("num_cavs", 1.0), 1.0),
            "cav.relative_distance_to_ego": _to_float(cav.get("relative_distance_to_ego", 0.0)),
            "cav.ego_demand_mean": _to_float(cav.get("ego_demand_mean", 0.0)),
            "rule.distance_proxy": _to_float(state.get("distance_proxy", max(_to_float(lidar.get("distant_voxel_ratio", 0.0)), distant_sparse))),
            "rule.clarity_proxy": _to_float(state.get("clarity_proxy", 0.5)),
        }
        return values

    def build_one(self, state: Dict[str, Any], normalize: bool = True, device=None) -> torch.Tensor:
        values = self.feature_dict(state)
        feature = torch.tensor([values.get(name, 0.0) for name in self.feature_names], dtype=self.dtype, device=device)
        if normalize:
            feature = self.normalize_tensor(feature)
        return feature

    def build_batch(self, states: List[Dict[str, Any]], normalize: bool = True, device=None) -> torch.Tensor:
        if not states:
            return torch.empty((0, self.dim), dtype=self.dtype, device=device)
        return torch.stack([self.build_one(state, normalize=normalize, device=device) for state in states], dim=0)

    def normalize_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.has_normalization():
            return feature
        mean = torch.tensor(self.feature_mean, dtype=feature.dtype, device=feature.device)
        std = torch.tensor(self.feature_std, dtype=feature.dtype, device=feature.device).clamp_min(1.0e-6)
        return (feature - mean) / std

    def to_json_dict(self) -> Dict[str, Any]:
        data = {"feature_names": self.feature_names}
        if self.feature_mean is not None:
            data["feature_mean"] = list(self.feature_mean)
        if self.feature_std is not None:
            data["feature_std"] = list(self.feature_std)
        return data

    def save_feature_names(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"feature_names": self.feature_names}, indent=2), encoding="utf-8")

    def save_norm(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_norm(cls, path: str, feature_names: Optional[List[str]] = None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = feature_names or data.get("feature_names") or DEFAULT_FEATURE_NAMES
        builder = cls(names)
        if "feature_mean" in data and "feature_std" in data:
            builder.set_normalization(data["feature_mean"], data["feature_std"])
        return builder

    @staticmethod
    def fit_normalization(features: torch.Tensor):
        if features.numel() == 0:
            raise ValueError("Cannot fit feature normalization on an empty tensor.")
        mean = features.float().mean(dim=0)
        std = features.float().std(dim=0, unbiased=False).clamp_min(1.0e-6)
        return mean.tolist(), std.tolist()
