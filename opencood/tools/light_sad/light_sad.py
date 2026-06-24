from typing import Tuple

from .config import LightSADConfig
from .sensor_stats import collect_light_sad_state


class LightSADDispatcher:
    def __init__(self, cfg=None):
        self.cfg = LightSADConfig.from_dict(cfg or {})

    def dispatch(self, data_dict: dict, record_len=None) -> dict:
        """
        Return a batch-level modality action and the state used to choose it.
        """
        state = collect_light_sad_state(data_dict)
        action, reason = self.decide_action(state)
        return {
            "action": action,
            "reason": reason,
            "state": state,
        }

    def decide_action(self, state: dict) -> Tuple[str, str]:
        cfg = self.cfg
        if cfg.force_action in {"L", "C", "LC"}:
            return cfg.force_action, f"force_action_{cfg.force_action}"

        lidar = state.get("lidar", {})
        camera = state.get("camera", {})
        network = state.get("network", {})

        lidar_valid = bool(lidar.get("valid", False))
        camera_valid = bool(camera.get("valid", False))
        lidar_points = int(lidar.get("num_points", 0))
        lidar_voxels = int(lidar.get("num_voxels", 0))
        brightness = float(camera.get("brightness", 0.0))
        blur_proxy = float(camera.get("blur_proxy", 0.0))
        bandwidth = float(network.get("bandwidth_mbps", 1000.0))
        rtt = float(network.get("rtt_ms", 0.0))

        lidar_good = lidar_valid and lidar_points >= cfg.lidar_good_points and lidar_voxels >= cfg.lidar_min_voxels
        lidar_weak = (not lidar_valid) or lidar_points < cfg.lidar_min_points or lidar_voxels < cfg.lidar_min_voxels
        camera_dark = camera_valid and brightness < cfg.camera_dark_thr
        camera_blurry = camera_valid and blur_proxy < cfg.camera_blur_thr
        camera_good = (
            camera_valid
            and brightness >= cfg.camera_good_brightness_thr
            and blur_proxy >= cfg.camera_blur_thr
        )
        network_bad = bandwidth < cfg.low_bandwidth_mbps or rtt > cfg.high_rtt_ms

        if not lidar_valid and camera_valid:
            return "C", "lidar_invalid_camera_available"
        if not camera_valid and lidar_valid:
            return "L", "camera_invalid_lidar_available"
        if network_bad:
            if lidar_good:
                return "L", "poor_network_good_lidar"
            if lidar_weak and camera_valid:
                return "C", "poor_network_weak_lidar_camera_available"
            return "L", "poor_network_fallback_lidar"
        if camera_dark:
            return "L", "camera_dark_lidar_preferred"
        if camera_blurry:
            return "L", "camera_blurry_lidar_preferred"
        if lidar_weak and camera_good:
            return "LC", "weak_lidar_camera_good"
        if lidar_good:
            return "L", "good_lidar_low_cost"
        return "LC", "default_multimodal"
