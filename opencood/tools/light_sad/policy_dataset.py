import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

try:
    from .feature_builder import LightSADFeatureBuilder
except ImportError:
    from opencood.tools.light_sad.feature_builder import LightSADFeatureBuilder


def read_oracle_records(paths) -> List[Dict[str, Any]]:
    if isinstance(paths, (list, tuple)):
        records = []
        for path in paths:
            records.extend(read_oracle_records(path))
        return records
    if isinstance(paths, str) and "," in paths and not Path(paths).exists():
        return read_oracle_records([x.strip() for x in paths.split(",") if x.strip()])

    path = Path(paths)
    if not path.exists():
        raise FileNotFoundError("Oracle policy dataset not found: %s" % path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("records", data) if isinstance(data, dict) else data
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    if suffix in {".pt", ".pth"}:
        data = torch.load(str(path), map_location="cpu")
        return data.get("records", data) if isinstance(data, dict) else data
    raise ValueError("Unsupported oracle dataset suffix: %s" % suffix)


def _label_to_index(label, action_set: List[str]) -> int:
    if isinstance(label, str):
        label = label.upper()
        if label not in action_set:
            raise ValueError("Unknown action label %s for action_set %s" % (label, action_set))
        return action_set.index(label)
    return int(label)


def _utility_vector(record: Dict[str, Any], action_set: List[str]) -> torch.Tensor:
    utility_dict = record.get("utility", None)
    values = []
    for action in action_set:
        if isinstance(utility_dict, dict) and action in utility_dict:
            values.append(float(utility_dict[action]))
        else:
            values.append(float(record.get("utility_%s" % action, 0.0)))
    return torch.tensor(values, dtype=torch.float32)


def _cost_vector(record: Dict[str, Any], action_set: List[str]) -> torch.Tensor:
    values = []
    compute = record.get("compute_cost_dict", {}) or {}
    comm = record.get("comm_cost_dict", {}) or {}
    latency = record.get("latency_cost_dict", {}) or {}
    total = record.get("cost", {}) or {}
    for action in action_set:
        if action in total:
            values.append(float(total[action]))
        else:
            values.append(float(compute.get(action, 0.0)) + float(comm.get(action, 0.0)) + float(latency.get(action, 0.0)))
    return torch.tensor(values, dtype=torch.float32)


class OraclePolicyDataset(Dataset):
    def __init__(
        self,
        path,
        action_set: Optional[Iterable[str]] = None,
        feature_builder: Optional[LightSADFeatureBuilder] = None,
        normalize: bool = True,
        min_utility_margin: float = 0.0,
    ):
        self.action_set = [str(x).upper() for x in (action_set or ["L", "C", "LC"])]
        self.normalize = bool(normalize)
        records = read_oracle_records(path)
        self.records = []
        for record in records:
            utilities = _utility_vector(record, self.action_set)
            if utilities.numel() > 1:
                top2 = utilities.topk(k=2).values
                margin = float(top2[0] - top2[1])
                if margin < float(min_utility_margin):
                    continue
            self.records.append(record)
        if not self.records:
            raise ValueError("No oracle records left after filtering: %s" % path)

        if feature_builder is None:
            feature_names = self.records[0].get("feature_names", None)
            feature_builder = LightSADFeatureBuilder(feature_names=feature_names)
        self.feature_builder = feature_builder

    def __len__(self) -> int:
        return len(self.records)

    def _feature(self, record: Dict[str, Any]) -> torch.Tensor:
        vector = record.get("feature_vector", None)
        if vector is not None:
            feature = torch.tensor(vector, dtype=torch.float32)
            if feature.numel() != self.feature_builder.dim:
                raise ValueError(
                    "Feature dim mismatch for sample %s: expected %d, got %d"
                    % (record.get("sample_id", "?"), self.feature_builder.dim, feature.numel())
                )
            if self.normalize:
                feature = self.feature_builder.normalize_tensor(feature)
            return feature
        state = record.get("state_dict", record.get("state", {}))
        return self.feature_builder.build_one(state, normalize=self.normalize)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        utilities = _utility_vector(record, self.action_set)
        label = record.get("label", None)
        if label is None:
            label = int(torch.argmax(utilities).item())
        label_idx = _label_to_index(label, self.action_set)
        return {
            "features": self._feature(record),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "utilities": utilities,
            "cost": _cost_vector(record, self.action_set),
            "metadata": record.get("metadata", {}),
        }

    def raw_feature_tensor(self) -> torch.Tensor:
        old = self.normalize
        self.normalize = False
        try:
            return torch.stack([self._feature(record) for record in self.records], dim=0)
        finally:
            self.normalize = old

    def class_counts(self) -> Dict[str, int]:
        counts = {action: 0 for action in self.action_set}
        for record in self.records:
            utilities = _utility_vector(record, self.action_set)
            label = record.get("label", int(torch.argmax(utilities).item()))
            counts[self.action_set[_label_to_index(label, self.action_set)]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts()
        total = float(sum(counts.values()))
        weights = []
        for action in self.action_set:
            weights.append(total / max(float(counts[action]) * len(self.action_set), 1.0))
        return torch.tensor(weights, dtype=torch.float32)

    def weighted_sampler(self) -> WeightedRandomSampler:
        class_weights = self.class_weights()
        sample_weights = []
        for idx in range(len(self)):
            label = int(self[idx]["label"].item())
            sample_weights.append(float(class_weights[label].item()))
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def collate_policy_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "utilities": torch.stack([item["utilities"] for item in batch], dim=0),
        "cost": torch.stack([item["cost"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }
