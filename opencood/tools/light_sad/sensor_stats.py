import numbers
from typing import Any, Dict, List, Optional, Tuple

import torch


def safe_numel(x) -> int:
    if x is None:
        return 0
    if hasattr(x, "numel"):
        return int(x.numel())
    if hasattr(x, "size") and not isinstance(x, (list, tuple, dict)):
        try:
            size = x.size
            if isinstance(size, numbers.Number):
                return int(size)
        except Exception:
            pass
    if hasattr(x, "shape"):
        total = 1
        for dim in x.shape:
            total *= int(dim)
        return int(total)
    try:
        return len(x)
    except TypeError:
        return 0


def record_len_to_list(record_len) -> List[int]:
    if record_len is None:
        return []
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def total_cavs(record_len) -> int:
    lengths = record_len_to_list(record_len)
    return int(sum(lengths)) if lengths else 1


def safe_mean(values: List[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def _to_float(value, default=0.0):
    if value is None:
        return float(default)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item") and safe_numel(value) == 1:
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def _default_lidar_stats(valid: bool = False) -> Dict[str, Any]:
    return {
        "num_points": 0,
        "num_voxels": 0,
        "mean_points_per_voxel": 0.0,
        "near_voxel_ratio": 0.0,
        "distant_voxel_ratio": 0.0,
        "distant_sparse_score": 0.0,
        "valid": bool(valid),
    }


def _default_camera_stats(valid: bool = False) -> Dict[str, Any]:
    return {
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_proxy": 0.0,
        "dark_score": 1.0 if not valid else 0.0,
        "valid": bool(valid),
    }


def _is_image_tensor(tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor) or tensor.dim() < 4:
        return False
    shape = list(tensor.shape)
    if shape[-1] < 8 or shape[-2] < 8:
        return False
    channel_like = False
    if tensor.dim() >= 3 and shape[-3] in (1, 3):
        channel_like = True
    if shape[-1] in (1, 3) and shape[-2] >= 8 and shape[-3] >= 8:
        channel_like = True
    return channel_like


def _first_image_tensor(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj if _is_image_tensor(obj) else None
    if isinstance(obj, dict):
        # In OpenCOOD camera inputs, "imgs" is the true image tensor. Avoid
        # rots/trans/intrinsics/extrinsics, which also are tensors but too small.
        if torch.is_tensor(obj.get("imgs", None)) and _is_image_tensor(obj["imgs"]):
            return obj["imgs"]
        for key in sorted(obj.keys()):
            tensor = _first_image_tensor(obj[key])
            if tensor is not None:
                return tensor
    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = _first_image_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _lidar_group_mode(voxel_coords, record_len) -> str:
    lengths = record_len_to_list(record_len)
    total = int(sum(lengths)) if lengths else 1
    if voxel_coords is None or safe_numel(voxel_coords) == 0:
        return "missing"
    coords = _to_tensor(voxel_coords)
    if coords is None or coords.dim() < 2 or coords.shape[1] < 1:
        return "missing"
    max_index = int(coords[:, 0].max().item()) if coords.shape[0] > 0 else -1
    # In the standard SiMO/OpenCOOD intermediate collate path, processed_lidar
    # is merged across all CAVs before pre_processor.collate_batch(). The first
    # coordinate column is therefore a flattened CAV index in [0, sum(record_len)).
    if max_index < total:
        return "flattened_cav"
    if lengths and max_index < len(lengths):
        return "batch_index"
    return "unknown"


def _stats_from_lidar_group(voxel_num_points, voxel_coords, mask) -> Dict[str, Any]:
    stats = _default_lidar_stats(False)
    if mask is None:
        return stats
    mask = mask.bool()
    num_voxels = int(mask.sum().item())
    if num_voxels <= 0:
        return stats

    stats["num_voxels"] = num_voxels
    if voxel_num_points is not None:
        pts = _to_tensor(voxel_num_points)
        num_points = int(pts[mask].sum().item()) if pts is not None and pts.shape[0] == mask.shape[0] else num_voxels
    else:
        num_points = num_voxels

    stats["num_points"] = int(num_points)
    stats["mean_points_per_voxel"] = float(num_points) / max(num_voxels, 1)

    coords = _to_tensor(voxel_coords)
    if coords is not None and coords.dim() >= 2 and coords.shape[1] >= 3:
        xy = coords[mask][:, -2:].float()
        if xy.shape[0] > 0:
            center = (xy.max(dim=0).values + xy.min(dim=0).values) / 2.0
            dist = torch.norm(xy - center, dim=1)
            max_dist = float(dist.max().item()) if dist.numel() > 0 else 0.0
            if max_dist > 1e-6:
                near = (dist <= 0.33 * max_dist).float().mean().item()
                distant = (dist >= 0.66 * max_dist).float().mean().item()
            else:
                near, distant = 1.0, 0.0
            density_score = min(stats["mean_points_per_voxel"] / 5.0, 1.0)
            stats["near_voxel_ratio"] = float(near)
            stats["distant_voxel_ratio"] = float(distant)
            stats["distant_sparse_score"] = float(distant * (1.0 - density_score))

    stats["valid"] = True
    return stats


def extract_lidar_stats_per_cav(data_dict: dict, record_len=None) -> Tuple[List[Dict[str, Any]], bool, str]:
    processed = data_dict.get("processed_lidar", None)
    total = total_cavs(record_len)
    if not isinstance(processed, dict):
        return [_default_lidar_stats(False) for _ in range(total)], False, "missing_processed_lidar"

    voxel_num_points = processed.get("voxel_num_points", None)
    voxel_features = processed.get("voxel_features", None)
    voxel_coords = processed.get("voxel_coords", None)
    coords = _to_tensor(voxel_coords)
    if coords is None and voxel_features is not None:
        global_stats = extract_lidar_stats(data_dict)
        return [global_stats.copy() for _ in range(total)], False, "missing_voxel_coords"

    mode = _lidar_group_mode(coords, record_len)
    if mode == "flattened_cav":
        stats = []
        index = coords[:, 0].long()
        for cav_idx in range(total):
            stats.append(_stats_from_lidar_group(voxel_num_points, coords, index == cav_idx))
        return stats, True, "flattened_cav"

    if mode == "batch_index":
        lengths = record_len_to_list(record_len)
        index = coords[:, 0].long()
        stats = []
        for batch_idx, cav_num in enumerate(lengths):
            sample_stats = _stats_from_lidar_group(voxel_num_points, coords, index == batch_idx)
            stats.extend([sample_stats.copy() for _ in range(cav_num)])
        return stats, False, "voxel_coords_batch_index_broadcast"

    global_stats = extract_lidar_stats(data_dict)
    return [global_stats.copy() for _ in range(total)], False, "fallback_global_lidar"


def extract_lidar_stats(data_dict: dict) -> dict:
    processed = data_dict.get("processed_lidar", None)
    if not isinstance(processed, dict):
        return _default_lidar_stats(False)
    voxel_coords = _to_tensor(processed.get("voxel_coords", None))
    if voxel_coords is None or safe_numel(voxel_coords) == 0:
        return _default_lidar_stats(False)
    mask = torch.ones((voxel_coords.shape[0],), dtype=torch.bool, device=voxel_coords.device)
    return _stats_from_lidar_group(processed.get("voxel_num_points", None), voxel_coords, mask)


def _camera_quality_from_tensor(img: torch.Tensor) -> Dict[str, Any]:
    stats = _default_camera_stats(False)
    if img is None or safe_numel(img) == 0:
        return stats

    img = img.detach().float()
    raw_min = float(img.min().item())
    raw_max = float(img.max().item())
    quality_img = img
    if raw_max > 2.0 and raw_min >= -1e-3:
        quality_img = img / 255.0

    unit_like = raw_min >= -0.05 and raw_max <= 1.5
    brightness = float(quality_img.mean().item()) if unit_like or raw_max > 2.0 else 0.5
    contrast = float(quality_img.std(unbiased=False).item())
    if quality_img.dim() >= 2 and quality_img.shape[-2] > 1 and quality_img.shape[-1] > 1:
        dh = torch.mean(torch.abs(quality_img[..., 1:, :] - quality_img[..., :-1, :]))
        dw = torch.mean(torch.abs(quality_img[..., :, 1:] - quality_img[..., :, :-1]))
        blur_proxy = float((dh + dw).item())
    else:
        blur_proxy = 0.0

    # Brightness is auxiliary. For normalized images with negative means, avoid
    # declaring them dark solely from the mean.
    dark_score = 0.0
    if unit_like or raw_max > 2.0:
        dark_score = max(0.0, min(1.0, (0.12 - brightness) / 0.12))
    if contrast < 1e-6 and blur_proxy < 1e-6:
        dark_score = max(dark_score, 0.8)

    stats.update({
        "brightness": brightness,
        "contrast": contrast,
        "blur_proxy": blur_proxy,
        "dark_score": float(dark_score),
        "valid": True,
    })
    return stats


def extract_camera_stats_per_cav(data_dict: dict, record_len=None) -> Tuple[List[Dict[str, Any]], bool, str]:
    total = total_cavs(record_len)
    image_inputs = data_dict.get("image_inputs", None)
    img = _first_image_tensor(image_inputs)
    if img is None:
        return [_default_camera_stats(False) for _ in range(total)], False, "missing_image_tensor"

    if img.shape[0] == total:
        return [_camera_quality_from_tensor(img[i]) for i in range(total)], True, "flattened_cav"

    lengths = record_len_to_list(record_len)
    if lengths and img.shape[0] == len(lengths):
        stats = []
        for batch_idx, cav_num in enumerate(lengths):
            sample_stats = _camera_quality_from_tensor(img[batch_idx])
            stats.extend([sample_stats.copy() for _ in range(cav_num)])
        return stats, False, "image_batch_index_broadcast"

    global_stats = _camera_quality_from_tensor(img)
    return [global_stats.copy() for _ in range(total)], False, "fallback_global_camera"


def extract_camera_stats(data_dict: dict) -> dict:
    img = _first_image_tensor(data_dict.get("image_inputs", None))
    return _camera_quality_from_tensor(img) if img is not None else _default_camera_stats(False)


def extract_network_stats(data_dict: dict) -> dict:
    network_state = data_dict.get("network_state", {}) or {}
    return {
        "bandwidth_mbps": _to_float(network_state.get("bandwidth_mbps", 1000.0), 1000.0),
        "rtt_ms": _to_float(network_state.get("rtt_ms", 0.0), 0.0),
        "packet_loss": _to_float(network_state.get("packet_loss", 0.0), 0.0),
        "queue_delay_ms": _to_float(network_state.get("queue_delay_ms", 0.0), 0.0),
    }


def extract_network_stats_per_cav(data_dict: dict, record_len=None) -> Tuple[List[Dict[str, Any]], bool, str]:
    total = total_cavs(record_len)
    per_cav = data_dict.get("network_state_per_cav", None)
    if isinstance(per_cav, list) and len(per_cav) >= total:
        return [dict(per_cav[i]) for i in range(total)], True, "network_state_per_cav"
    return [extract_network_stats(data_dict).copy() for _ in range(total)], False, "default_good_network_broadcast"


def _merge_global(per_cav_stats: List[Dict[str, Any]], default_fn) -> Dict[str, Any]:
    if not per_cav_stats:
        return default_fn(False)
    keys = per_cav_stats[0].keys()
    merged = {}
    for key in keys:
        values = [item.get(key) for item in per_cav_stats]
        if isinstance(values[0], bool):
            merged[key] = any(bool(v) for v in values)
        elif isinstance(values[0], numbers.Number):
            merged[key] = safe_mean([float(v) for v in values])
        else:
            merged[key] = values[0]
    if "num_points" in merged:
        merged["num_points"] = int(sum(int(item.get("num_points", 0)) for item in per_cav_stats))
    if "num_voxels" in merged:
        merged["num_voxels"] = int(sum(int(item.get("num_voxels", 0)) for item in per_cav_stats))
        merged["mean_points_per_voxel"] = float(merged.get("num_points", 0)) / max(merged["num_voxels"], 1)
    return merged


def collect_light_sad_state(data_dict: dict, record_len=None, per_cav: bool = False) -> dict:
    lidar_pc, lidar_ok, lidar_reason = extract_lidar_stats_per_cav(data_dict, record_len)
    camera_pc, camera_ok, camera_reason = extract_camera_stats_per_cav(data_dict, record_len)
    network_pc, network_ok, network_reason = extract_network_stats_per_cav(data_dict, record_len)
    history = data_dict.get("light_sad_history", None) or {}

    state = {
        "lidar": _merge_global(lidar_pc, _default_lidar_stats),
        "camera": _merge_global(camera_pc, _default_camera_stats),
        "network": extract_network_stats(data_dict),
        "history": history,
        "per_cav": {
            "lidar": lidar_pc,
            "camera": camera_pc,
            "network": network_pc,
        },
        "per_cav_valid": bool(lidar_ok and camera_ok),
        "fallback_reasons": {
            "lidar": lidar_reason,
            "camera": camera_reason,
            "network": network_reason,
        },
    }
    return state
