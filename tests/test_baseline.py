"""Small offline checks for classification/regression baseline infrastructure."""

import unittest

import numpy as np
import pandas as pd
import torch

from src.data import make_splits
from src.engine import (
    branch_cross_entropy, branch_mse, evaluate_regression, make_loader,
)


class _IdentityEnsemble(torch.nn.Module):
    def forward(self, x_num=None, x_cat=None):
        # The numerical input is already a standardized regression target.
        return x_num[:, None, :1].expand(-1, 2, -1)


class BaselineTests(unittest.TestCase):
    def test_regression_splits_are_disjoint_and_complete(self):
        y = pd.Series(np.arange(100, dtype=float))
        splits = make_splits(y, seed=42, regression=True)
        combined = np.concatenate((splits.train, splits.val, splits.test))
        self.assertEqual(len(splits.train), 70)
        self.assertEqual(len(splits.val), 15)
        self.assertEqual(len(splits.test), 15)
        np.testing.assert_array_equal(np.sort(combined), np.arange(100))

    def test_classification_splits(self):
        labels = pd.Series([0] * 50 + [1] * 50)
        splits = make_splits(labels, seed=42)
        for part in (splits.train, splits.val, splits.test):
            self.assertEqual(set(labels.iloc[part]), {0, 1})

    def test_branch_regression_mse(self):
        pred = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]])
        truth = torch.tensor([2.0, 2.0])
        self.assertAlmostEqual(branch_mse(pred, truth).item(), 1.5)

    def test_branch_classification_ce(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        labels = torch.tensor([0])
        loss = branch_cross_entropy(logits, labels)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2), labels.repeat_interleave(2)
        )
        self.assertAlmostEqual(loss.item(), expected.item())

    def test_regression_metric_inverse_standardization(self):
        y_std = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        num = y_std[:, None]
        cat = np.empty((4, 0), dtype=np.int64)
        loader = make_loader(num, cat, y_std, batch_size=2, shuffle=False)
        scores = evaluate_regression(
            _IdentityEnsemble(), loader, torch.device('cpu'),
            target_mean=2.0, target_std=3.0,
        )
        self.assertAlmostEqual(scores['rmse'], 0.0)
        self.assertAlmostEqual(scores['mae'], 0.0)
        self.assertAlmostEqual(scores['r2'], 1.0)


if __name__ == '__main__':
    unittest.main()
