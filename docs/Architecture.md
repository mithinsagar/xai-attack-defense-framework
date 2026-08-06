# System Architecture

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

## Overall Pipeline

```
[ Phase 1 ]  Project Setup & Data Preprocessing
        |
        v
[ Phase 2 ]  Base Model Training  (LR / XGBoost / MLP)
        |
        v
[ Phase 3 ]  Explanation Generation  (SHAP / LIME / IG / Permutation)
        |
        v
[ Phase 4 ]  Attack Generation  (Attacks 1 - 4)
        |
        v
[ Phase 5 ]  Few-Shot Vulnerability Analysis  (Prototypical Network)
        |
        v
[ Phase 6 ]  Defense Architectures  (Defenses 1 - 4)
        |
        v
[  Results  ]  Aggregated CSVs, plots, and summary report
```

## Technology Stack

| Category           | Tool / Library                     |
|--------------------|------------------------------------|
| Deep Learning      | PyTorch                            |
| ML Framework       | scikit-learn                       |
| Gradient Boosting  | XGBoost                            |
| Explainability     | SHAP, LIME, Captum (Integrated Gradients) |
| Visualization      | Matplotlib, Seaborn                |
| Data Manipulation  | NumPy, Pandas                      |
| Configuration      | YAML, JSON                         |
| Testing            | Pytest                             |

## Module Map

```
config.py           - Central configuration (paths, hyperparameters)
main.py             - CLI orchestrator for all six phases

data/               - Raw + processed dataset handling
models/             - Model classes (Logistic, XGBoost, MLP, RobustMLP)
explainers/         - SHAP, LIME, IG, Permutation Importance
attacks/            - Attack 1-4 implementations + runner
defenses/           - Defense 1-4 implementations + trainer dispatcher
fewshot/            - Episode sampler, Prototypical Network, trainer
metrics/            - Drift metric, classification eval, comparison tables
training/           - End-to-end training entrypoints
visualization/      - Common plot helpers
utils/              - I/O, logging, seeding, constants
tests/              - Unit tests
notebooks/          - Original per-phase Jupyter notebooks
scripts/            - Shell / Python helpers (run_all, downloads, report)
docs/               - Extended documentation and paper assets
```
