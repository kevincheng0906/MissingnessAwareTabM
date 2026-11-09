"""Re-evaluate an existing TabM checkpoint on its unchanged test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from tabm import TabM

from .data import encode_labels, load_raw_dataset
from .engine import evaluate, make_loader, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    results = json.loads((args.run_dir / 'metrics.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(args.run_dir / 'best.pt', map_location='cpu', weights_only=True)
    prep = joblib.load(args.run_dir / 'preprocessor.joblib')  # Only load your own trusted files.
    with np.load(args.run_dir / 'splits.npz') as splits:
        train_idx, test_idx = splits['train'], splits['test']
    x, raw_y = load_raw_dataset(results['dataset'])
    y, encoder = encode_labels(raw_y, train_idx)
    if [str(v) for v in encoder.classes_] != results['class_names']:
        raise RuntimeError('Dataset label mapping changed; cannot safely evaluate.')
    x_num, x_cat = prep.transform(x.iloc[test_idx])
    loader = make_loader(x_num, x_cat, y[test_idx], args.batch_size, shuffle=False)
    device = select_device(args.device)
    model = TabM.make(**checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    metrics = evaluate(model, loader, device)
    print(json.dumps({'best_epoch': checkpoint['best_epoch'], 'test': metrics}, indent=2))


if __name__ == '__main__':
    main()
