import torch


def action_to_runtime_mask(action: str, batch_size: int, cav_num: int, device) -> dict:
    """
    Convert a Light-SAD action to camera/lidar runtime masks for LAMMA.
    """
    cam = torch.ones((batch_size, cav_num), device=device)
    lidar = torch.ones((batch_size, cav_num), device=device)

    if action == "L":
        cam.zero_()
    elif action == "C":
        lidar.zero_()
    elif action == "LC":
        pass
    else:
        raise ValueError(f"Unsupported Light-SAD action: {action}")

    return {
        "camera": cam,
        "lidar": lidar,
    }
