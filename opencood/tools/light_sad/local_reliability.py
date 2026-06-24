from typing import Any, Dict, List, Optional

import torch

from .sensor_stats import _to_tensor, record_len_to_list, safe_numel, total_cavs


def _empty_map(grid_size: int, value: float = 0.0):
    return torch.full((grid_size, grid_size), float(value))


def _coords_to_reliability_map(coords, grid_size: int):
    if coords is None or safe_numel(coords) == 0:
        return _empty_map(grid_size, 0.0)
    xy = coords[:, -2:].float()
    if xy.numel() == 0:
        return _empty_map(grid_size, 0.0)
    xy_min = xy.min(dim=0).values
    xy_max = xy.max(dim=0).values
    span = torch.clamp(xy_max - xy_min, min=1.0)
    norm = torch.clamp((xy - xy_min) / span, min=0.0, max=0.999)
    indices = (norm * grid_size).long()
    grid = torch.zeros((grid_size, grid_size), device=xy.device)
    grid.index_put_((indices[:, 0], indices[:, 1]), torch.ones(indices.shape[0], device=xy.device), accumulate=True)
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid.cpu()


def _groups_from_coords(voxel_coords, record_len):
    coords = _to_tensor(voxel_coords)
    total = total_cavs(record_len)
    if coords is None or coords.dim() < 2 or coords.shape[0] == 0:
        return [None for _ in range(total)], False
    first = coords[:, 0].long()
    max_idx = int(first.max().item())
    if max_idx < total:
        return [coords[first == i] for i in range(total)], True
    lengths = record_len_to_list(record_len)
    if lengths and max_idx < len(lengths):
        groups = []
        for sample_idx, cav_num in enumerate(lengths):
            sample_coords = coords[first == sample_idx]
            groups.extend([sample_coords for _ in range(cav_num)])
        return groups, False
    return [None for _ in range(total)], False


def build_local_reliability(data_dict: dict, record_len=None, camera_stats_per_cav: Optional[List[Dict[str, Any]]] = None, grid_size: int = 16) -> List[Dict[str, Any]]:
    """
    Build coarse BEV reliability summaries.

    This is a MoME-style local reliability proxy only. It does not change
    Pyramid Fusion weights and does not import MoME decoder/AQR/MED modules.
    """
    processed = data_dict.get("processed_lidar", None)
    voxel_coords = processed.get("voxel_coords", None) if isinstance(processed, dict) else None
    groups, reliable_split = _groups_from_coords(voxel_coords, record_len)
    total = len(groups)
    outputs = []
    for idx in range(total):
        lidar_map = _coords_to_reliability_map(groups[idx], grid_size)
        cam_stats = camera_stats_per_cav[idx] if camera_stats_per_cav and idx < len(camera_stats_per_cav) else {}
        camera_quality = 0.0
        if cam_stats.get("valid", False):
            camera_quality = max(0.0, min(1.0, float(cam_stats.get("contrast", 0.0)) + float(cam_stats.get("blur_proxy", 0.0))))
            camera_quality *= (1.0 - min(float(cam_stats.get("dark_score", 0.0)), 1.0))
        camera_map = _empty_map(grid_size, camera_quality)
        low_lidar_region_ratio = float((lidar_map < 0.15).float().mean().item())
        camera_reliable_flag = bool(camera_quality >= 0.08)
        prefer = torch.full((grid_size, grid_size), 2, dtype=torch.long)
        prefer[(lidar_map >= 0.35) & (camera_map < 0.08)] = 0
        prefer[(lidar_map < 0.15) & (camera_map >= 0.08)] = 1
        outputs.append({
            "lidar_reliability_map": lidar_map,
            "camera_reliability_map": camera_map,
            "local_prefer_map": prefer,
            "summary": {
                "low_lidar_region_ratio": low_lidar_region_ratio,
                "camera_reliable_flag": camera_reliable_flag,
                "reliable_split": bool(reliable_split),
            },
        })
    return outputs
