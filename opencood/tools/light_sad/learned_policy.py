from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLightSADPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_set: Optional[Iterable[str]] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_set = [str(x).upper() for x in (action_set or ["L", "C", "LC"])]
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(max(num_layers, 1))
        self.dropout = float(dropout)
        self.layer_norm = bool(layer_norm)

        layers = []
        last_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(last_dim, self.hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            last_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, len(self.action_set))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.head(self.backbone(x.float()))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> Dict[str, Any]:
        logits = self.forward(x)
        probs = F.softmax(logits / max(float(temperature), 1.0e-6), dim=-1)
        conf, index = probs.max(dim=-1)
        actions = [self.action_set[int(i)] for i in index.detach().cpu().tolist()]
        top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            margins = top2[:, 0]
        else:
            margins = top2[:, 0] - top2[:, 1]
        return {
            "actions": actions,
            "action": actions[0] if len(actions) == 1 else actions,
            "prob": conf.detach().cpu(),
            "probs": probs.detach().cpu(),
            "logits": logits.detach().cpu(),
            "margin": margins.detach().cpu(),
        }

    def model_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "action_set": self.action_set,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "layer_norm": self.layer_norm,
        }


def save_policy_checkpoint(
    policy: LearnedLightSADPolicy,
    path: str,
    feature_names: List[str],
    feature_mean: Optional[List[float]] = None,
    feature_std: Optional[List[float]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy.state_dict(),
        "model_config": policy.model_config(),
        "action_set": policy.action_set,
        "feature_names": list(feature_names),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_config": train_config or {},
        "metrics": metrics or {},
    }
    torch.save(payload, str(path))


def load_policy_checkpoint(
    path: str,
    map_location: str = "cpu",
    expected_feature_names: Optional[List[str]] = None,
) -> Tuple[LearnedLightSADPolicy, Dict[str, Any]]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError("Learned Light-SAD checkpoint not found: %s" % ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid Light-SAD checkpoint: expected dict, got %s" % type(ckpt))

    feature_names = ckpt.get("feature_names")
    model_config = dict(ckpt.get("model_config", {}))
    action_set = ckpt.get("action_set", model_config.get("action_set", ["L", "C", "LC"]))
    input_dim = int(model_config.get("input_dim", len(feature_names or [])))
    if input_dim <= 0:
        raise ValueError("Cannot infer policy input_dim from checkpoint %s" % ckpt_path)

    if expected_feature_names is not None and feature_names is not None and list(expected_feature_names) != list(feature_names):
        expected = list(expected_feature_names)
        actual = list(feature_names)
        raise ValueError(
            "Light-SAD feature names mismatch. expected=%s actual=%s" % (expected, actual)
        )

    policy = LearnedLightSADPolicy(
        input_dim=input_dim,
        action_set=action_set,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        layer_norm=bool(model_config.get("layer_norm", True)),
    )
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state is None:
        state = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    policy.load_state_dict(state)
    policy.eval()
    return policy, ckpt
