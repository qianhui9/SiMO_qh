import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .feature_builder import LightSADFeatureBuilder
    from .learned_policy import LearnedLightSADPolicy, save_policy_checkpoint
    from .policy_dataset import OraclePolicyDataset, collate_policy_batch
except ImportError:
    from opencood.tools.light_sad.feature_builder import LightSADFeatureBuilder
    from opencood.tools.light_sad.learned_policy import LearnedLightSADPolicy, save_policy_checkpoint
    from opencood.tools.light_sad.policy_dataset import OraclePolicyDataset, collate_policy_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train learned Light-SAD MLP policy.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--beta_cost", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--class_balance", default="none", choices=["none", "loss", "sampler"])
    parser.add_argument("--min_utility_margin", type=float, default=0.0)
    parser.add_argument("--action_set", default="L,C,LC")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def macro_f1(confusion: torch.Tensor) -> float:
    scores = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(sum(scores) / max(len(scores), 1))


@torch.no_grad()
def evaluate(model, loader, device, action_set):
    model.eval()
    total = 0
    correct = 0
    top2_correct = 0
    confusion = torch.zeros((len(action_set), len(action_set)), dtype=torch.long)
    pred_counts = torch.zeros(len(action_set), dtype=torch.long)
    label_counts = torch.zeros(len(action_set), dtype=torch.long)
    regrets = []
    pred_utils = []
    oracle_utils = []
    confidences = []

    for batch in loader:
        x = batch["features"].to(device)
        labels = batch["label"].to(device)
        utilities = batch["utilities"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        top2 = logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices
        total += labels.numel()
        correct += int((pred == labels).sum().item())
        top2_correct += int((top2 == labels.unsqueeze(-1)).any(dim=-1).sum().item())
        for y, p in zip(labels.detach().cpu(), pred.detach().cpu()):
            confusion[int(y), int(p)] += 1
            pred_counts[int(p)] += 1
            label_counts[int(y)] += 1
        oracle = utilities.max(dim=-1).values
        pred_utility = utilities.gather(1, pred.unsqueeze(1)).squeeze(1)
        oracle_utils.extend(oracle.detach().cpu().tolist())
        pred_utils.extend(pred_utility.detach().cpu().tolist())
        regrets.extend((oracle - pred_utility).detach().cpu().tolist())
        confidences.extend(probs.max(dim=-1).values.detach().cpu().tolist())

    total = max(total, 1)
    action_distribution = {action_set[i]: int(pred_counts[i].item()) for i in range(len(action_set))}
    label_distribution = {action_set[i]: int(label_counts[i].item()) for i in range(len(action_set))}
    return {
        "classification_accuracy": float(correct / total),
        "top2_accuracy": float(top2_correct / total),
        "macro_f1": macro_f1(confusion),
        "oracle_utility": float(sum(oracle_utils) / max(len(oracle_utils), 1)),
        "predicted_utility": float(sum(pred_utils) / max(len(pred_utils), 1)),
        "utility_regret": float(sum(regrets) / max(len(regrets), 1)),
        "action_distribution": action_distribution,
        "label_distribution": label_distribution,
        "confusion_matrix": confusion.tolist(),
        "confidence_mean": float(sum(confidences) / max(len(confidences), 1)),
        "confidence_std": float(torch.tensor(confidences).std(unbiased=False).item()) if confidences else 0.0,
    }


def main():
    args = parse_args()
    action_set = [x.strip().upper() for x in args.action_set.split(",") if x.strip()]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = OraclePolicyDataset(
        args.train_path,
        action_set=action_set,
        normalize=False,
        min_utility_margin=args.min_utility_margin,
    )
    raw_features = base_dataset.raw_feature_tensor()
    mean, std = LightSADFeatureBuilder.fit_normalization(raw_features)
    builder = LightSADFeatureBuilder(base_dataset.feature_builder.feature_names, mean, std)
    builder.save_norm(str(save_dir / "feature_norm.json"))

    train_dataset = OraclePolicyDataset(
        args.train_path,
        action_set=action_set,
        feature_builder=builder,
        normalize=True,
        min_utility_margin=args.min_utility_margin,
    )
    val_dataset = OraclePolicyDataset(
        args.val_path or args.train_path,
        action_set=action_set,
        feature_builder=builder,
        normalize=True,
        min_utility_margin=args.min_utility_margin,
    )

    sampler = train_dataset.weighted_sampler() if args.class_balance == "sampler" else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_policy_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_policy_batch,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = LearnedLightSADPolicy(
        input_dim=builder.dim,
        action_set=action_set,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        layer_norm=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weight = train_dataset.class_weights().to(device) if args.class_balance == "loss" else None
    history = []
    best_regret = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            utilities = batch["utilities"].to(device)
            cost = batch["cost"].to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, labels, weight=class_weight)
            soft_target = torch.softmax(utilities / max(args.temperature, 1.0e-6), dim=-1)
            kl = F.kl_div(
                F.log_softmax(logits / max(args.temperature, 1.0e-6), dim=-1),
                soft_target,
                reduction="batchmean",
            ) * (args.temperature ** 2)
            expected_cost = (torch.softmax(logits, dim=-1) * cost).sum(dim=-1).mean()
            loss = ce + args.alpha_kl * kl + args.beta_cost * expected_cost
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach().cpu().item()))

        train_metrics = evaluate(model, train_loader, device, action_set)
        val_metrics = evaluate(model, val_loader, device, action_set)
        train_loss = float(sum(running) / max(len(running), 1))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            "[Light-SAD policy] epoch=%d loss=%.6f val_acc=%.4f val_regret=%.6f dist=%s"
            % (epoch, train_loss, val_metrics["classification_accuracy"], val_metrics["utility_regret"], val_metrics["action_distribution"])
        )

        save_policy_checkpoint(
            model,
            str(save_dir / "last.pth"),
            builder.feature_names,
            builder.feature_mean,
            builder.feature_std,
            train_config=vars(args),
            metrics=val_metrics,
        )
        if best_regret is None or val_metrics["utility_regret"] < best_regret:
            best_regret = val_metrics["utility_regret"]
            save_policy_checkpoint(
                model,
                str(save_dir / "best.pth"),
                builder.feature_names,
                builder.feature_mean,
                builder.feature_std,
                train_config=vars(args),
                metrics=val_metrics,
            )

    (save_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (save_dir / "val_metrics.json").write_text(json.dumps(history[-1]["val"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
