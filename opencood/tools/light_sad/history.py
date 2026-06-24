from typing import Any, Dict, Optional

import torch


class HistoryConfidenceBuffer:
    """
    Lightweight ego/batch-level confidence memory for Light-SAD v1.

    This is not a tracker. It only stores recent detection score summaries so
    the next frame can choose a more conservative modality action when the
    previous frame looked uncertain.
    """

    def __init__(self, topk: int = 20, stale_limit: int = 3):
        self.topk = int(topk)
        self.stale_limit = int(stale_limit)
        self.reset()

    def reset(self):
        self.last_mean_score = 0.0
        self.last_topk_mean_score = 0.0
        self.last_num_detections = 0
        self.stale_frames = self.stale_limit + 1
        self.valid = False

    def update(self, scores: Optional[Any]):
        values = self._to_1d_tensor(scores)
        if values is None or values.numel() == 0:
            self.stale_frames += 1
            self.valid = self.stale_frames <= self.stale_limit and self.last_num_detections > 0
            return self.get_state()

        values = values.detach().float().cpu()
        self.last_num_detections = int(values.numel())
        self.last_mean_score = float(values.mean().item())
        k = min(self.topk, int(values.numel()))
        self.last_topk_mean_score = float(torch.topk(values, k).values.mean().item()) if k > 0 else 0.0
        self.stale_frames = 0
        self.valid = True
        return self.get_state()

    def step_without_update(self):
        self.stale_frames += 1
        self.valid = self.stale_frames <= self.stale_limit and self.last_num_detections > 0
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_mean_score": float(self.last_mean_score),
            "last_topk_mean_score": float(self.last_topk_mean_score),
            "last_num_detections": int(self.last_num_detections),
            "stale_frames": int(self.stale_frames),
            "valid": bool(self.valid and self.stale_frames <= self.stale_limit),
        }

    @staticmethod
    def _to_1d_tensor(scores):
        if scores is None:
            return None
        if torch.is_tensor(scores):
            return scores.reshape(-1)
        try:
            return torch.as_tensor(scores).reshape(-1)
        except Exception:
            return None
