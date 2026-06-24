import torch

from .light_sad import LightSADDispatcher
from .runtime_mask import action_to_runtime_mask


def _lidar_case(num_voxels, points_per_voxel):
    voxel_num_points = torch.full((num_voxels,), points_per_voxel, dtype=torch.int32)
    return {
        "voxel_features": torch.ones((num_voxels, 4, 4)),
        "voxel_coords": torch.zeros((num_voxels, 4), dtype=torch.int32),
        "voxel_num_points": voxel_num_points,
    }


def _camera_case(kind="normal"):
    if kind == "dark":
        imgs = torch.zeros((1, 2, 3, 32, 32)) + 0.04
    else:
        torch.manual_seed(7)
        imgs = torch.rand((1, 2, 3, 32, 32)) * 0.6 + 0.2
    return {"imgs": imgs}


def _data_dict(lidar, camera, network=None):
    return {
        "processed_lidar": lidar,
        "image_inputs": camera,
        "network_state": network or {},
    }


def _run_case(name, dispatcher, data_dict, expected):
    result = dispatcher.dispatch(data_dict)
    action = result["action"]
    reason = result["reason"]
    print(f"[Light-SAD verify] case={name} action={action} reason={reason}")
    if isinstance(expected, set):
        assert action in expected, f"{name}: expected {expected}, got {action}"
    else:
        assert action == expected, f"{name}: expected {expected}, got {action}"


def main():
    dispatcher = LightSADDispatcher({"enabled": True})

    _run_case(
        "good_lidar",
        dispatcher,
        _data_dict(_lidar_case(900, 10), _camera_case("normal")),
        "L",
    )
    _run_case(
        "weak_lidar",
        dispatcher,
        _data_dict(_lidar_case(200, 3), _camera_case("normal")),
        "LC",
    )
    _run_case(
        "dark_camera",
        dispatcher,
        _data_dict(_lidar_case(600, 8), _camera_case("dark")),
        "L",
    )
    _run_case(
        "force_camera",
        LightSADDispatcher({"enabled": True, "policy": "force", "force_action": "C"}),
        _data_dict(_lidar_case(900, 10), _camera_case("normal")),
        "C",
    )

    mask = action_to_runtime_mask("L", batch_size=1, cav_num=3, device=torch.device("cpu"))
    assert mask["camera"].sum().item() == 0
    assert mask["lidar"].sum().item() == 3
    print("Light-SAD verification passed.")


if __name__ == "__main__":
    main()
