"""Shared, unmodified TabM training/evaluation semantics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


def select_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def make_loader(
    num: np.ndarray,
    cat: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(num),
        torch.from_numpy(cat),
        torch.from_numpy(y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def forward_tabm(model, numeric, categorical, device: torch.device):
    x_num = numeric.to(device) if numeric.shape[1] else None
    x_cat = categorical.to(device) if categorical.shape[1] else None
    return model(x_num, x_cat)


def branch_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average CE over samples AND branches, as in the official TabM training."""
    b, k, c = logits.shape
    return F.cross_entropy(logits.reshape(b * k, c), target.repeat_interleave(k))


def branch_mse(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average per-branch MSE on train-standardized regression targets."""
    if predictions.ndim != 3 or predictions.shape[-1] != 1:
        raise ValueError(f'Regression output must be (B, K, 1), got {predictions.shape}')
    pred = predictions.squeeze(-1)
    return F.mse_loss(pred, target[:, None].expand_as(pred))


@torch.inference_mode()
def predict_regression(
    model, loader: DataLoader, device: torch.device,
    target_mean: float, target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average branch predictions and invert the train-fitted target scaling."""
    model.eval()
    predictions, labels = [], []
    for x_num, x_cat, y in loader:
        outputs = forward_tabm(model, x_num, x_cat, device)
        preds = outputs.squeeze(-1).mean(dim=1)
        predictions.append(preds.cpu().numpy())
        labels.append(y.numpy())
    preds = np.concatenate(predictions) * target_std + target_mean
    truth = np.concatenate(labels) * target_std + target_mean
    return preds, truth


def evaluate_regression(
    model, loader: DataLoader, device: torch.device,
    target_mean: float, target_std: float,
) -> dict[str, float]:
    preds, truth = predict_regression(
        model, loader, device, target_mean, target_std
    )
    return {
        'rmse': float(np.sqrt(mean_squared_error(truth, preds))),
        'mae': float(mean_absolute_error(truth, preds)),
        'r2': float(r2_score(truth, preds)),
    }


@torch.inference_mode()
def predict_probabilities(model, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities, labels = [], []
    for x_num, x_cat, y in loader:
        logits = forward_tabm(model, x_num, x_cat, device)
        # Classification ensembles average probabilities, NOT logits.
        probs = logits.softmax(dim=-1).mean(dim=1)
        probabilities.append(probs.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def calculate_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    scores = {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'log_loss': float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
    }
    if probs.shape[1] == 2 and np.unique(y).size == 2:
        scores['roc_auc'] = float(roc_auc_score(y, probs[:, 1]))
    return scores


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    probs, y = predict_probabilities(model, loader, device)
    return calculate_metrics(probs, y)
