import numbers

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


def _first_tensor(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if torch.is_tensor(obj.get("imgs", None)):
            return obj["imgs"]
        for value in obj.values():
            tensor = _first_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = _first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def extract_lidar_stats(data_dict: dict) -> dict:
    """
    Extract cheap LiDAR quality statistics from data_dict['processed_lidar'].
    """
    processed = data_dict.get("processed_lidar", None)
    stats = {
        "num_points": 0,
        "num_voxels": 0,
        "mean_points_per_voxel": 0.0,
        "valid": False,
    }

    if not isinstance(processed, dict):
        return stats

    voxel_num_points = processed.get("voxel_num_points", None)
    voxel_features = processed.get("voxel_features", None)
    voxel_coords = processed.get("voxel_coords", None)

    num_voxels = 0
    if voxel_coords is not None and hasattr(voxel_coords, "shape") and len(voxel_coords.shape) > 0:
        num_voxels = int(voxel_coords.shape[0])
    elif voxel_features is not None and hasattr(voxel_features, "shape") and len(voxel_features.shape) > 0:
        num_voxels = int(voxel_features.shape[0])

    num_points = 0
    voxel_num_points_tensor = _to_tensor(voxel_num_points)
    if voxel_num_points_tensor is not None and safe_numel(voxel_num_points_tensor) > 0:
        num_points = int(voxel_num_points_tensor.sum().item())
    elif voxel_features is not None:
        num_points = num_voxels

    stats["num_points"] = int(num_points)
    stats["num_voxels"] = int(num_voxels)
    stats["mean_points_per_voxel"] = float(num_points) / max(num_voxels, 1)
    stats["valid"] = num_voxels > 0 or num_points > 0
    return stats


def extract_camera_stats(data_dict: dict) -> dict:
    """
    Extract cheap camera quality statistics from data_dict['image_inputs'].
    """
    image_inputs = data_dict.get("image_inputs", None)
    img = _first_tensor(image_inputs)
    stats = {
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_proxy": 0.0,
        "valid": False,
    }
    if img is None or safe_numel(img) == 0:
        return stats

    img = img.detach().float()
    brightness = img.mean()
    if brightness.item() > 1.5:
        img = img / 255.0
        brightness = img.mean()

    contrast = img.std(unbiased=False)
    blur_proxy = torch.tensor(0.0, device=img.device)
    if img.dim() >= 2:
        dh = torch.mean(torch.abs(img[..., 1:, :] - img[..., :-1, :])) if img.shape[-2] > 1 else 0.0
        dw = torch.mean(torch.abs(img[..., :, 1:] - img[..., :, :-1])) if img.shape[-1] > 1 else 0.0
        blur_proxy = dh + dw

    stats["brightness"] = float(brightness.item())
    stats["contrast"] = float(contrast.item())
    stats["blur_proxy"] = _to_float(blur_proxy)
    stats["valid"] = True
    return stats


def extract_network_stats(data_dict: dict) -> dict:
    """
    Read optional network state, defaulting to a healthy link.
    """
    network_state = data_dict.get("network_state", {}) or {}
    return {
        "bandwidth_mbps": _to_float(network_state.get("bandwidth_mbps", 1000.0), 1000.0),
        "rtt_ms": _to_float(network_state.get("rtt_ms", 0.0), 0.0),
        "packet_loss": _to_float(network_state.get("packet_loss", 0.0), 0.0),
        "queue_delay_ms": _to_float(network_state.get("queue_delay_ms", 0.0), 0.0),
    }


def collect_light_sad_state(data_dict: dict) -> dict:
    return {
        "lidar": extract_lidar_stats(data_dict),
        "camera": extract_camera_stats(data_dict),
        "network": extract_network_stats(data_dict),
    }
