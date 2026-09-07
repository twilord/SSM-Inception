"""Train and evaluate SSM-Inception on the reported DSADS subject split."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from ssm_inception import SSMInception
from ssm_inception.data import load_dsads, prepare_subject_independent_data


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def make_model(config: dict) -> SSMInception:
    dataset, model = config["dataset"], config["model"]
    expected_fixed_configuration = {
        "inception_channels": [64, 128],
        "inception_kernels": [3, 5, 7, 9],
    }
    for key, expected in expected_fixed_configuration.items():
        if model.get(key) != expected:
            raise ValueError(
                f"this minimal implementation requires model.{key}={expected}"
            )
    return SSMInception(
        num_channels=dataset["channels"],
        num_classes=dataset["classes"],
        embedding_dim=model["embedding_dim"],
        state_size=model["s4d_state_size"],
        s4d_dropout=model["s4d_dropout"],
    )


def loader(x, y, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


@torch.inference_mode()
def predict(model, data_loader, device):
    true_parts, predicted_parts = [], []
    model.eval()
    for x, y in data_loader:
        predicted = model(x.to(device)).argmax(dim=1).cpu().numpy()
        true_parts.append(y.numpy())
        predicted_parts.append(predicted)
    return np.concatenate(true_parts), np.concatenate(predicted_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dsads.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = args.data_root or config["dataset"]["root"]
    if data_root is None:
        parser.error("set dataset.root in YAML or pass --data-root")
    training = config["training"]
    seed = int(training["seed"])
    deterministic = bool(training["deterministic"])
    set_seed(seed, deterministic)

    x, y, subjects = load_dsads(
        data_root,
        median_kernel_size=config["preprocessing"]["median_kernel_size"],
        median_boundary_mode=config["preprocessing"]["median_boundary_mode"],
    )
    prepared = prepare_subject_independent_data(
        x,
        y,
        subjects,
        epsilon=float(config["preprocessing"]["epsilon"]),
        development_subjects=config["split"]["development_subjects"],
        test_subjects=config["split"]["test_subjects"],
    )
    dev_x, dev_y, dev_subjects = prepared["development"]
    test_x, test_y, test_subjects = prepared["test"]
    mean, std = prepared["normalization"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)
    split_manifest = {
        "development_subjects": config["split"]["development_subjects"],
        "test_subjects": config["split"]["test_subjects"],
        "development_windows": int(len(dev_x)),
        "test_windows": int(len(test_x)),
        "validation": "window-level KFold within the development pool",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    folds = int(config["split"]["folds"])
    splitter = KFold(
        n_splits=folds,
        shuffle=bool(config["split"]["kfold_shuffle"]),
        random_state=int(config["split"]["kfold_random_state"]),
    )
    set_seed(seed, deterministic)
    initial_state = copy.deepcopy(make_model(config).state_dict())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_results, prediction_parts = [], []

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(dev_x)):
        set_seed(seed + fold, deterministic)
        model = make_model(config).to(device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = loader(
            dev_x[train_indices],
            dev_y[train_indices],
            int(training["batch_size"]),
            True,
            seed + fold,
        )
        validation_loader = loader(
            dev_x[validation_indices],
            dev_y[validation_indices],
            int(training["batch_size"]),
            False,
            seed,
        )
        test_loader = loader(
            test_x, test_y, int(training["batch_size"]), False, seed
        )

        best_accuracy, best_epoch, best_state = -1.0, -1, None
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            val_true, val_pred = predict(model, validation_loader, device)
            val_accuracy = accuracy_score(val_true, val_pred)
            if val_accuracy > best_accuracy:
                best_accuracy = float(val_accuracy)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"fold={fold} epoch={epoch:03d} val_acc={val_accuracy:.6f} "
                f"best={best_accuracy:.6f}",
                flush=True,
            )

        model.load_state_dict(best_state)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(best_state, fold_dir / "checkpoint.pt")
        test_true, test_pred = predict(model, test_loader, device)
        accuracy = float(accuracy_score(test_true, test_pred))
        macro_f1 = float(
            f1_score(
                test_true,
                test_pred,
                labels=np.arange(int(config["dataset"]["classes"])),
                average="macro",
                zero_division=0,
            )
        )
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "test_accuracy": accuracy,
                "test_macro_f1": macro_f1,
            }
        )
        prediction_parts.append(
            {
                "y_true": test_true,
                "y_pred": test_pred,
                "subject_id": test_subjects.copy(),
                "fold": np.full(len(test_true), fold, dtype=np.int64),
            }
        )

    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            key: np.concatenate([part[key] for part in prediction_parts])
            for key in prediction_parts[0]
        },
    )
    accuracies = np.asarray([row["test_accuracy"] for row in fold_results])
    macro_f1s = np.asarray([row["test_macro_f1"] for row in fold_results])
    result = {
        "config": config,
        "actual_device": str(device),
        "protocol": split_manifest,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "folds": fold_results,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=1)),
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_std": float(macro_f1s.std(ddof=1)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
