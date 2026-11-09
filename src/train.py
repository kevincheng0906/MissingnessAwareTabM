"""Original TabM classification/regression baseline with reproducible splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from tabm import TabM

from .data import TabularPreprocessor, encode_labels, load_raw_dataset, make_splits
from .engine import (
    branch_cross_entropy,
    branch_mse,
    evaluate,
    evaluate_regression,
    forward_tabm,
    make_loader,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the ORIGINAL TabM baseline.')
    parser.add_argument('--dataset', choices=['adult', 'breast_cancer', 'california_housing'], default='adult')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.0003)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--n-blocks', type=int, default=3)
    parser.add_argument('--d-block', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--run-dir', type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError('--epochs and --patience must be positive')
    set_seed(args.seed)
    device = select_device(args.device)
    run_dir = args.run_dir or Path('runs') / f'{args.dataset}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    regression = args.dataset == 'california_housing'
    task_type = 'regression' if regression else 'classification'
    x, raw_y = load_raw_dataset(args.dataset)
    splits = make_splits(raw_y, seed=args.split_seed, regression=regression)
    np.savez(run_dir / 'splits.npz', train=splits.train, val=splits.val, test=splits.test)
    preprocessor = TabularPreprocessor().fit(x.iloc[splits.train])
    joblib.dump(preprocessor, run_dir / 'preprocessor.joblib')
    if regression:
        raw_values = raw_y.to_numpy(dtype=np.float32)
        target_mean = float(raw_values[splits.train].mean())
        target_std = float(raw_values[splits.train].std())
        if target_std <= 0:
            raise ValueError('Training targets must have nonzero standard deviation')
        y = ((raw_values - target_mean) / target_std).astype(np.float32)
        label_encoder = None
    else:
        y, label_encoder = encode_labels(raw_y, splits.train)
        target_mean, target_std = None, None

    arrays = {}
    for part in ('train', 'val', 'test'):
        idx = getattr(splits, part)
        num, cat = preprocessor.transform(x.iloc[idx])
        arrays[part] = (num, cat, y[idx])

    train_loader = make_loader(*arrays['train'], args.batch_size, shuffle=True)
    val_loader = make_loader(*arrays['val'], args.batch_size, shuffle=False)
    test_loader = make_loader(*arrays['test'], args.batch_size, shuffle=False)

    model_kwargs = {
        'n_num_features': len(preprocessor.num_columns),
        'cat_cardinalities': preprocessor.cat_cardinalities,
        'd_out': 1 if regression else int(len(label_encoder.classes_)),
        'k': args.k,
        'n_blocks': args.n_blocks,
        'd_block': args.d_block,
        'dropout': args.dropout,
    }
    model = TabM.make(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    stale = 0
    print(f'Dataset={args.dataset}, train/val/test={[len(splits.train), len(splits.val), len(splits.test)]}')
    if regression:
        print(f'Device={device}, branches={model.k}, task=regression, target mean/std={target_mean:.4f}/{target_std:.4f}')
    else:
        print(f'Device={device}, branches={model.k}, classes={label_encoder.classes_.tolist()}')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_seen = 0.0, 0
        for x_num, x_cat, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_tabm(model, x_num, x_cat, device)
            loss = branch_mse(logits, labels) if regression else branch_cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_seen += len(labels)
        train_loss = total_loss / total_seen
        val_metrics = (
            evaluate_regression(model, val_loader, device, target_mean, target_std)
            if regression else evaluate(model, val_loader, device)
        )
        loss_name = 'train_branch_mse' if regression else 'train_branch_ce'
        row = {'epoch': epoch, loss_name: train_loss, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        if regression:
            print(f'Epoch {epoch:03d} | train branch MSE {train_loss:.4f} | '
                  f'val RMSE {val_metrics["rmse"]:.4f} | '
                  f'val MAE {val_metrics["mae"]:.4f} | '
                  f'val R2 {val_metrics["r2"]:.4f}')
        else:
            print(f'Epoch {epoch:03d} | train branch CE {train_loss:.4f} | '
                  f'val logloss {val_metrics["log_loss"]:.4f} | '
                  f'val accuracy {val_metrics["accuracy"]:.4f} | '
                  f'val macro-F1 {val_metrics["macro_f1"]:.4f}')
        early_stop_metric = 'rmse' if regression else 'log_loss'
        if val_metrics[early_stop_metric] < best_val_loss - 1e-6:
            best_val_loss = val_metrics[early_stop_metric]
            best_epoch = epoch
            stale = 0
            torch.save({'state_dict': model.state_dict(), 'model_kwargs': model_kwargs,
                        'best_epoch': best_epoch, 'seed': args.seed, 'task_type': task_type}, run_dir / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print(f'Early stopping at epoch {epoch}; best epoch was {best_epoch}.')
                break

    with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    score_fn = (
        lambda loader: evaluate_regression(model, loader, device, target_mean, target_std)
        if regression else evaluate(model, loader, device)
    )
    val_metrics = score_fn(val_loader)
    test_metrics = score_fn(test_loader)  # Test evaluated ONLY after selecting best epoch.
    result = {
        'dataset': args.dataset,
        'task_type': task_type,
        'split_seed': args.split_seed,
        'train_seed': args.seed,
        'best_epoch': best_epoch,
        'n_train': len(splits.train),
        'n_val': len(splits.val),
        'n_test': len(splits.test),
        'feature_count': {'num': len(preprocessor.num_columns), 'cat': len(preprocessor.cat_columns)},
        'class_names': None if regression else [str(x) for x in label_encoder.classes_],
        'target_transform': {'mean': target_mean, 'std': target_std} if regression else None,
        'model_kwargs': model_kwargs,
        'val': val_metrics,
        'test': test_metrics,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nBest checkpoint:', run_dir / 'best.pt')
    print('Validation:', val_metrics)
    print('Test:      ', test_metrics)


if __name__ == '__main__':
    main()
