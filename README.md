# Missingness-Aware TabM

**Missingness-Aware TabM: Robust Tabular Learning via Branch-Specific Masking and Dynamic Gating**

NYCU Deep Learning (Fall 2026), Master's final project.

## Phase 1 — Original TabM Baseline

A reproducible classification and regression baseline using the official [TabM package](https://github.com/yandex-research/tabm). This phase **does not use** branch-specific masking or dynamic gating.

### Dataset and methods

- **Adult Income** (OpenML ID 1590): primary binary-classification dataset (first download requires internet).
- **California Housing** (scikit-learn): primary regression benchmark (20,640 samples, 8 numerical features; first download may require internet).
- **Breast Cancer** (scikit-learn): quick smoke-test dataset, not a paper-scale benchmark.
- Fixed **70% train / 15% validation / 15% test** splits, controlled by `--split-seed` (stratified for classification; random for regression).
- Fit median imputation, standardization, and categorical encoders **only on train**. Unseen categories get a reserved index.
- Classification: **cross-entropy per ensemble branch**, averaged over branches. Evaluation averages **softmax probabilities**, not logits.
- Regression: **MSE per ensemble branch** on targets standardized using *train-only* statistics. At evaluation, inverse-transform each prediction and compute RMSE, MAE and R² in the original target units.
- AdamW, gradient clipping, early stopping and checkpoint selection by validation **log loss** (classification) or **RMSE** (regression).
- Evaluate the held-out test split only after selecting the best checkpoint.
- Classification results include accuracy, macro-F1, log loss and binary ROC-AUC; regression results include RMSE, MAE and R².

### Structure

```text
MissingnessAwareTabM/
├── src/
│   ├── __init__.py
│   ├── data.py            # dataset loading and train-only preprocessing
│   ├── engine.py          # branch-wise loss, ensemble predictions, metrics
│   ├── train.py           # baseline training and checkpointing
│   └── evaluate.py        # re-evaluate saved checkpoint
├── requirements.txt
├── .gitignore
└── README.md
```

### Environment

Python 3.10–3.12 recommended. Install a CUDA-enabled PyTorch build appropriate for your GPU if needed. Then:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Offline sanity checks

```bash
python -m unittest discover -s tests -v
```

These tests cover classification/regression loss shapes, label splits, and regression inverse scaling without requiring a dataset download.

### Smoke test

```bash
python -m src.train --dataset breast_cancer --epochs 5 --patience 3 --k 4 --d-block 64
python -m src.evaluate --run-dir runs/breast_cancer_seed42
```

### Baseline experiments

```bash
python -m src.train --dataset adult --epochs 150 --patience 20 --k 32
python -m src.evaluate --run-dir runs/adult_seed42

python -m src.train --dataset california_housing --epochs 150 --patience 20 --k 32
python -m src.evaluate --run-dir runs/california_housing_seed42
```

Use `python -m src.train --help` for all settings. The output directory `runs/<dataset>_seed<seed>/` contains:

- `best.pt`: best TabM checkpoint and architecture arguments.
- `splits.npz`: fixed train, validation and test indices.
- `preprocessor.joblib`: train-fitted imputation, scalers and encoders.
- `history.csv`: per-epoch validation metrics.
- `metrics.json`: training setup, selected epoch, validation and test metrics.

The baseline supports two task types but introduces no new TabM modules. It is a **controlled implementation of original TabM**; it does not claim to exactly reproduce paper scores, which depend on benchmark-specific feature embeddings and hyperparameter tuning.

### Planned ablations

1. Add missingness simulation after the fixed data split.
2. Add branch-specific feature masking to TabM's supported `(batch_size, k, n_features)` input interface.
3. Add an input/missingness-aware dynamic gating network.
4. Evaluate no modification / masking only / gating only / combined approach across multiple seeds.

## References

- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (ICLR 2025)](https://arxiv.org/abs/2410.24210)
- [TabM official source](https://github.com/yandex-research/tabm)
