"""Reproducible dataset loading and training-only preprocessing for TabM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


@dataclass
class DatasetSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_raw_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load rows in a stable order. Adult requires an internet connection once."""
    if name == 'adult':
        x, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    elif name == 'breast_cancer':
        x, y = load_breast_cancer(as_frame=True, return_X_y=True)
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return x.reset_index(drop=True), pd.Series(y).reset_index(drop=True)


def make_splits(labels: pd.Series, seed: int = 42) -> DatasetSplits:
    indices = np.arange(len(labels), dtype=np.int64)
    train, holdout = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    val, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        stratify=labels.iloc[holdout],
    )
    return DatasetSplits(train=train, val=val, test=test)


class TabularPreprocessor:
    """Fit on train rows only; categorical indices reserve 0 for unseen values."""

    def fit(self, x: pd.DataFrame) -> 'TabularPreprocessor':
        self.columns = x.columns.tolist()
        self.num_columns = [c for c in self.columns if pd.api.types.is_numeric_dtype(x[c])]
        self.cat_columns = [c for c in self.columns if c not in self.num_columns]
        self.num_imputer = SimpleImputer(strategy='median') if self.num_columns else None
        self.num_scaler = StandardScaler() if self.num_columns else None
        self.cat_encoder = (
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            if self.cat_columns else None
        )
        if self.num_columns:
            x_num = self.num_imputer.fit_transform(x[self.num_columns])
            self.num_scaler.fit(x_num)
        if self.cat_columns:
            self.cat_encoder.fit(self._prepare_cat(x))
        return self

    def _prepare_cat(self, x: pd.DataFrame) -> pd.DataFrame:
        cols = x[self.cat_columns].astype('object')
        return cols.replace('?', np.nan).fillna('__MISSING__').astype(str)

    @property
    def cat_cardinalities(self) -> list[int]:
        if self.cat_encoder is None:
            return []
        return [len(categories) + 1 for categories in self.cat_encoder.categories_]

    def transform(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = x[self.columns]
        if self.num_columns:
            nums = self.num_imputer.transform(x[self.num_columns])
            nums = self.num_scaler.transform(nums).astype(np.float32)
        else:
            nums = np.empty((len(x), 0), dtype=np.float32)
        if self.cat_columns:
            # Encoder maps unseen categories to -1, which becomes reserved index 0.
            cats = (self.cat_encoder.transform(self._prepare_cat(x)) + 1).astype(np.int64)
        else:
            cats = np.empty((len(x), 0), dtype=np.int64)
        return nums, cats


def encode_labels(
    raw_y: pd.Series, train_idx: np.ndarray
) -> tuple[np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder().fit(raw_y.iloc[train_idx])
    y = label_encoder.transform(raw_y).astype(np.int64)
    return y, label_encoder
