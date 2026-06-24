from typing import List, Optional

import torch


def _record_len_to_list(record_len) -> List[int]:
    if record_len is None:
        return []
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def _validate_action(action: str) -> str:
    if action not in {"L", "C", "LC"}:
        raise ValueError("Unsupported Light-SAD action: %s" % action)
    return action


def expand_actions(actions, total_cavs: int) -> List[str]:
    if isinstance(actions, str):
        parts = [x.strip() for x in actions.split(",") if x.strip()]
    else:
        parts = list(actions or [])
    if not parts:
        parts = ["LC"]
    parts = [_validate_action(str(x)) for x in parts]
    expanded = []
    for idx in range(total_cavs):
        expanded.append(parts[idx % len(parts)])
    return expanded


def action_to_runtime_mask(action, batch_size: Optional[int] = None, cav_num: Optional[int] = None, device=None, record_len=None) -> dict:
    """
    Convert a global action or flattened per-CAV actions to LAMMA masks.

    Global action:
      action_to_runtime_mask("L", B, N, device)

    Per-CAV action:
      action_to_runtime_mask(["L", "LC", "C"], record_len=record_len, device=device)
    """
    lengths = _record_len_to_list(record_len)
    is_per_cav = isinstance(action, (list, tuple)) or (isinstance(action, str) and "," in action)

    if is_per_cav:
        total = sum(lengths) if lengths else int(batch_size or 1) * int(cav_num or 1)
        actions = expand_actions(action, total)
        if lengths:
            batch_size = len(lengths)
            cav_num = max(lengths) if lengths else total
        else:
            batch_size = int(batch_size or 1)
            cav_num = int(cav_num or max(total, 1))
            lengths = [cav_num for _ in range(batch_size)]
        cam = torch.zeros((batch_size, cav_num), device=device)
        lidar = torch.zeros((batch_size, cav_num), device=device)
        flat_idx = 0
        for b, length in enumerate(lengths):
            for n in range(min(length, cav_num)):
                act = actions[flat_idx]
                cam[b, n] = 1.0 if "C" in act else 0.0
                lidar[b, n] = 1.0 if "L" in act else 0.0
                flat_idx += 1
        return {"camera": cam, "lidar": lidar}

    action = _validate_action(action)
    cam = torch.ones((batch_size, cav_num), device=device)
    lidar = torch.ones((batch_size, cav_num), device=device)
    if action == "L":
        cam.zero_()
    elif action == "C":
        lidar.zero_()
    return {"camera": cam, "lidar": lidar}
