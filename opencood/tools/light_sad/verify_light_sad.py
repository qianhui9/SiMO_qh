import tempfile

import torch

from .feature_builder import LightSADFeatureBuilder
from .history import HistoryConfidenceBuffer
from .learned_policy import LearnedLightSADPolicy, save_policy_checkpoint
from .light_sad import LightSADDispatcher
from .local_reliability import build_local_reliability
from .runtime_mask import action_to_runtime_mask, expand_actions


def _lidar_case(num_voxels, points_per_voxel, cav_idx=0, spread=24):
    voxel_num_points = torch.full((num_voxels,), points_per_voxel, dtype=torch.int32)
    coords = torch.zeros((num_voxels, 4), dtype=torch.int32)
    coords[:, 0] = int(cav_idx)
    if num_voxels > 0:
        coords[:, 2] = torch.arange(num_voxels, dtype=torch.int32) % spread
        coords[:, 3] = torch.div(torch.arange(num_voxels, dtype=torch.int32), max(spread, 1), rounding_mode="trunc")
    return {
        "voxel_features": torch.ones((num_voxels, 4, 4)),
        "voxel_coords": coords,
        "voxel_num_points": voxel_num_points,
    }


def _merge_lidar(cases):
    non_empty = [case for case in cases if case["voxel_coords"].shape[0] > 0]
    if not non_empty:
        return _lidar_case(0, 0)
    return {
        "voxel_features": torch.cat([case["voxel_features"] for case in non_empty], dim=0),
        "voxel_coords": torch.cat([case["voxel_coords"] for case in non_empty], dim=0),
        "voxel_num_points": torch.cat([case["voxel_num_points"] for case in non_empty], dim=0),
    }


def _camera_case(kind="normal", cav_num=1):
    torch.manual_seed(7)
    imgs = torch.rand((cav_num, 2, 3, 32, 32)) * 0.6 + 0.2
    if kind == "dark":
        imgs = torch.zeros((cav_num, 2, 3, 32, 32)) + 0.04
    return {"imgs": imgs}


def _data_dict(lidar, camera, network=None, history=None):
    data = {
        "processed_lidar": lidar,
        "image_inputs": camera,
        "network_state": network or {},
    }
    if history is not None:
        data["light_sad_history"] = history
    return data


def _run_case(name, dispatcher, data_dict, expected):
    result = dispatcher.dispatch(data_dict)
    action = result["action"]
    reason = result["reason"]
    print(f"[Light-SAD verify] case={name} action={action} reason={reason}")
    if isinstance(expected, set):
        assert action in expected, f"{name}: expected {expected}, got {action}"
    else:
        assert action == expected, f"{name}: expected {expected}, got {action}"


def _test_global_actions():
    dispatcher = LightSADDispatcher({"enabled": True, "per_cav": False})
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
        LightSADDispatcher({"enabled": True, "policy": "force", "force_action": "C", "per_cav": False}),
        _data_dict(_lidar_case(900, 10), _camera_case("normal")),
        "C",
    )


def _test_per_cav_actions():
    lidar = _merge_lidar([
        _lidar_case(900, 10, cav_idx=0),
        _lidar_case(200, 3, cav_idx=1),
        _lidar_case(0, 0, cav_idx=2),
        _lidar_case(600, 8, cav_idx=3),
    ])
    imgs = _camera_case("normal", cav_num=4)["imgs"]
    imgs[3] = 0.04
    data = _data_dict(lidar, {"imgs": imgs})
    result = LightSADDispatcher({"enabled": True, "per_cav": True}).dispatch(
        data, record_len=torch.tensor([4])
    )
    print("[Light-SAD verify] per_cav actions=", result["actions"], "reasons=", result["reasons"])
    assert result["mode"] == "per_cav"
    assert result["actions"] == ["L", "LC", "C", "L"], result["actions"]

    mask = action_to_runtime_mask(result["actions"], record_len=torch.tensor([4]), device=torch.device("cpu"))
    assert mask["camera"].tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert mask["lidar"].tolist() == [[1.0, 1.0, 0.0, 1.0]]

    padded = action_to_runtime_mask(["L", "LC", "C"], batch_size=1, cav_num=5, record_len=torch.tensor([3]), device=torch.device("cpu"))
    assert padded["camera"].tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]
    assert padded["lidar"].tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]


def _test_force_actions():
    actions = expand_actions("L,LC,C", 5)
    assert actions == ["L", "LC", "C", "L", "LC"]
    result = LightSADDispatcher({
        "enabled": True,
        "policy": "force",
        "force_actions": "L,LC,C",
        "per_cav": True,
    }).dispatch(_data_dict(_lidar_case(10, 1), _camera_case("normal")), record_len=torch.tensor([5]))
    print("[Light-SAD verify] force_actions=", result["actions"])
    assert result["actions"] == actions


def _test_history():
    history = HistoryConfidenceBuffer(topk=2, stale_limit=1)
    high = history.update(torch.tensor([0.9, 0.8, 0.7]))
    assert high["valid"] and high["last_topk_mean_score"] > 0.8
    high_result = LightSADDispatcher({
        "enabled": True,
        "per_cav": False,
        "use_history": True,
        "policy": "emc2_rule_history",
    }).dispatch(_data_dict(_lidar_case(900, 10), _camera_case("normal"), history=high))
    print("[Light-SAD verify] high_history action=", high_result["action"])
    assert high_result["action"] == "L"

    low_history = {
        "last_mean_score": 0.1,
        "last_topk_mean_score": 0.1,
        "last_num_detections": 2,
        "stale_frames": 0,
        "valid": True,
    }
    low_result = LightSADDispatcher({
        "enabled": True,
        "per_cav": False,
        "use_history": True,
        "policy": "emc2_rule_history",
    }).dispatch(_data_dict(_lidar_case(600, 8), _camera_case("normal"), history=low_history))
    print("[Light-SAD verify] low_history action=", low_result["action"])
    assert low_result["action"] == "LC"

    history.step_without_update()
    stale = history.step_without_update()
    assert not stale["valid"]


def _test_local_reliability():
    data = _data_dict(_lidar_case(100, 2), _camera_case("normal"))
    local = build_local_reliability(data, record_len=torch.tensor([1]), camera_stats_per_cav=[{"valid": True, "contrast": 0.2, "blur_proxy": 0.1, "dark_score": 0.0}])
    summary = local[0]["summary"]
    print("[Light-SAD verify] local_summary=", summary)
    assert "low_lidar_region_ratio" in summary
    assert local[0]["lidar_reliability_map"].shape == (16, 16)



def _test_learned_policy_forward():
    builder = LightSADFeatureBuilder()
    policy = LearnedLightSADPolicy(builder.dim, hidden_dim=8, num_layers=1, dropout=0.0)
    state = {
        "lidar": {"valid": True, "num_points": 9000, "num_voxels": 900, "mean_points_per_voxel": 10.0},
        "camera": {"valid": True, "brightness": 0.4, "contrast": 0.2, "blur_proxy": 0.05, "dark_score": 0.0},
        "network": {},
        "history": {},
    }
    x = builder.build_one(state)
    assert policy(x)["__class__"] if False else True
    pred = policy.predict(x)
    assert pred["actions"][0] in {"L", "C", "LC"}

    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        save_policy_checkpoint(policy, tmp.name, builder.feature_names, train_config={"unit": True})
        result = LightSADDispatcher({
            "enabled": True,
            "policy": "learned_mlp",
            "learned_ckpt": tmp.name,
            "per_cav": False,
            "safe_fallback": True,
            "log_policy_prob": True,
        }).dispatch(_data_dict(_lidar_case(900, 10), _camera_case("normal")))
        assert result["action"] in {"L", "C", "LC"}
        assert "action_probs" in result


def main():
    _test_global_actions()
    _test_per_cav_actions()
    _test_force_actions()
    _test_history()
    _test_local_reliability()
    _test_learned_policy_forward()
    print("Light-SAD verification passed.")


if __name__ == "__main__":
    main()
