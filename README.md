# XAI Attack and Defense Framework with Few-Shot Learning

A comprehensive research framework for studying the robustness of Explainable AI
(XAI) methods against adversarial explanation manipulation attacks in
cybersecurity domains. The framework trains multiple base models across three
real-world security datasets, generates explanations via four XAI methods,
executes four families of explanation attacks, quantifies manipulation using a
formal Explanation Drift metric, analyses amplified vulnerability under
few-shot learning, and finally proposes four progressively stronger defense
architectures culminating in a Hybrid Defense that reduces explanation drift by
up to 91 percent while preserving prediction accuracy.

---

## Author

**Mithin Sagar S**
GitHub: <https://github.com/mithinsagar>

Course: Deep Learning 
Programme: B.Tech Computer Science and Engineering Specialisation in Artificial Intelligence and Machine Learning
Institution: Vellore Institute of Technology, Chennai

Collaborators on the original technical report: Gokul Ram K, Kishore A G.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Key Contributions](#key-contributions)
3. [Datasets](#datasets)
4. [System Architecture](#system-architecture)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Base Model Performance](#base-model-performance)
9. [Attack Framework](#attack-framework)
10. [Explanation Drift Metric](#explanation-drift-metric)
11. [Few-Shot Vulnerability](#few-shot-vulnerability)
12. [Defense Architectures](#defense-architectures)
13. [Results](#results)
14. [Documentation](#documentation)
15. [Citation](#citation)
16. [License](#license)

---

## Motivation

Modern machine learning systems used for phishing detection, network intrusion
detection, and online-fraud detection are frequently black boxes. Explainable
AI methods such as SHAP, LIME, and Integrated Gradients were introduced to
open these black boxes and give security analysts a defensible rationale for
each prediction. However, the explanations themselves can be adversarially
manipulated: a carefully crafted perturbation of the input can leave the
model's prediction untouched while completely shifting which features the
explanation highlights. An analyst who trusts a manipulated explanation may
dismiss real threats or chase phantom ones.

This project provides a reproducible, end-to-end pipeline for studying that
threat, quantifying it, and defending against it.

---

## Key Contributions

1. Multi-dataset, multi-model evaluation across three cybersecurity datasets
   and three learning paradigms (linear, gradient-boosted trees, deep
   neural network).
2. Four post-hoc explanation methods integrated in a single interface: SHAP,
   LIME, Integrated Gradients (Captum), and Permutation Importance.
3. Four adversarial explanation attacks with formal threat model.
4. A principled Explanation Drift metric with additional variants (Max Drift,
   Top-k Drift, Rank Correlation Drift).
5. Few-shot vulnerability study using Prototypical Networks.
6. Four defense architectures: Stability Training, Adversarial Training,
   Explanation Regularization, and a Hybrid Defense achieving up to 91.1
   percent drift reduction.

---

## Datasets

Three publicly available cybersecurity datasets are used. Download the raw
CSVs from the sources below and place them in `./data/raw/` (see
`data/README.md`).

| Dataset                          | Samples  | Features (used) | Task                      | Source |
|----------------------------------|----------|------------------|---------------------------|--------|
| PhiUSIIL Phishing URL Dataset    | 235,795  | 56 (51)          | Phishing vs Legitimate    | UCI Machine Learning Repository |
| Cybersecurity Intrusion Dataset  | 9,537    | 11 (10)          | Attack vs Normal          | Kaggle |
| Fraudulent Online Shops Dataset  | 1,140    | 26 (24)          | Fraud vs Legitimate       | Mendeley Data |

**Dataset download links:**

- PhiUSIIL Phishing URL Dataset:
  <https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- Cybersecurity Intrusion Detection Dataset:
  <https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset>
- Fraudulent Online Shops Dataset:
  <https://data.mendeley.com/datasets/m7xtkx7g5m/1>

Feature engineering note: `URLSimilarityIndex` is removed from the phishing
dataset because it exhibits near-constant behavior for the phishing class,
which would trivialize the classification task and produce misleading
explanations.

---

## System Architecture

The pipeline is organized into six sequential phases, each of which is also
available as a standalone Jupyter notebook in `notebooks/`.

```
Phase 1  ->  Project Setup and Data Preprocessing
Phase 2  ->  Base Model Training (Logistic, XGBoost, MLP)
Phase 3  ->  Explanation Generation (SHAP, LIME, IG, Permutation)
Phase 4  ->  Attack Generation (4 attacks)
Phase 5  ->  Few-Shot Vulnerability Analysis (Prototypical Network)
Phase 6  ->  Defense Architectures (4 defenses)
```

---

## Repository Structure

```
xai-attack-defense-framework/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
├── config.py
├── main.py
├── config/                (settings.yaml, model/attack/defense configs)
├── data/                  (loader, preprocessor, dataset info)
├── models/                (Logistic, XGBoost, MLP, RobustMLP, FewShot encoder)
├── explainers/            (SHAP, LIME, IG, Permutation wrappers)
├── attacks/               (4 attack strategies)
├── defenses/              (4 defense strategies + trainer)
├── metrics/               (Drift metric, evaluation, comparison)
├── fewshot/               (Episode sampler, Prototypical net, trainer)
├── training/              (End-to-end training scripts)
├── visualization/         (Plot helpers)
├── utils/                 (I/O, logging, seeds, constants)
├── tests/                 (Unit tests)
├── notebooks/             (Original Jupyter notebooks per phase)
├── scripts/               (Shell scripts, dataset download helpers)
├── docs/                  (Extended documentation, report, presentation)
├── results/               (Generated CSV result tables)
├── figures/               (Generated plots)
└── logs/                  (Training logs, configs)
```

---

## Installation

```bash
git clone https://github.com/mithinsagar/xai-attack-defense-framework.git
cd xai-attack-defense-framework

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.9 or newer is recommended. A CUDA-capable GPU is optional but
speeds up neural network and Integrated Gradients steps.

---

## Quick Start

```bash
# 1. Download the three datasets manually from the links above
#    and place the CSVs under data/raw/

# 2. Run the full pipeline end to end
python main.py --phase all

# Or run individual phases
python main.py --phase preprocess
python main.py --phase train_base
python main.py --phase explain
python main.py --phase attack
python main.py --phase fewshot
python main.py --phase defense
```

Interactive exploration is available via the notebooks in `notebooks/`.

---

## Base Model Performance

| Dataset  | Model               | Accuracy | F1 Score |
|----------|---------------------|----------|----------|
| Phishing | Logistic Regression | 0.9999   | 0.9999   |
| Phishing | XGBoost             | 1.0000   | 1.0000   |
| Phishing | Neural Network (MLP)| 0.9945   | 0.9952   |
| IDS      | Logistic Regression | 0.7159   | 0.6679   |
| IDS      | XGBoost             | 0.8863   | 0.8543   |
| IDS      | Neural Network (MLP)| 0.8643   | 0.8279   |
| Fraud    | Logistic Regression | 0.9649   | 0.9652   |
| Fraud    | XGBoost             | 0.9781   | 0.9780   |
| Fraud    | Neural Network (MLP)| 0.8728   | 0.8835   |

---

## Attack Framework

Formal threat model (white-box):

```
maximise ||E(x) - E(x + delta)||_1
subject to  f(x) = f(x + delta),   ||delta||_inf <= epsilon
```

| # | Attack                                | Target Features        | Epsilon |
|---|---------------------------------------|------------------------|---------|
| 1 | Top-Feature Perturbation              | Top-3 SHAP features    | 0.05    |
| 2 | Prediction-Stable Manipulation        | Bottom-3 SHAP features | 0.10    |
| 3 | Gradient-Based Attack                 | Top-3 gradient features| 0.05    |
| 4 | Targeted Explanation Attack           | Top-5 SHAP/IG features | 0.25    |

Observed attack strength ranking (mean drift, descending):

```
Attack 3 (Gradient)  >  Attack 4 (Targeted)  >  Attack 2 (Low)  >  Attack 1 (Top)
```

---

## Explanation Drift Metric

The Explanation Drift `D(x, epsilon)` is the mean absolute difference between
the per-feature attribution vectors before and after perturbation:

```
D(x, epsilon) = (1 / d) * sum_i |E_i(x) - E_i(x + delta)|
```

Additional variants: Max Drift, Top-k Drift, Rank Correlation Drift.
Implementations live in `metrics/drift_metric.py`.

---

## Few-Shot Vulnerability

A Prototypical Network encoder is trained on 50 samples per class from the
Phishing dataset (see `fewshot/`). The four attacks are then re-applied to
the few-shot model. Observed increase in explanation drift compared to the
full-data model:

| Attack               | Full Model Drift | Few-Shot Drift |
|----------------------|------------------|----------------|
| Attack 1 (Top)       | 0.0230           | 0.0412         |
| Attack 2 (Low)       | 0.0185           | 0.0356         |
| Attack 3 (Gradient)  | 0.0387           | 0.0623         |
| Attack 4 (Targeted)  | 0.0251           | 0.0498         |

Few-shot models exhibit 60 to 80 percent higher explanation drift.

---

## Defense Architectures

| Defense                    | Mechanism                                   | Mean Drift | Reduction |
|----------------------------|---------------------------------------------|------------|-----------|
| Baseline (no defense)      | -                                           | 0.038795   | -         |
| Defense 1: Stability       | Noise + output-consistency loss             | 0.003960   | 89.8%     |
| Defense 2: Adversarial     | Cross-entropy on adversarial samples        | 0.008447   | 78.2%     |
| Defense 3: Explanation Reg | IG attribution consistency loss             | 0.011453   | 70.5%     |
| Defense 4: Hybrid          | Stability + Adversarial + Smoothness        | **0.003462** | **91.1%** |

Defense 4 is the recommended production configuration.

---

## Results

All numerical results generated by the pipeline are written to `results/`:

- `baseline_results.csv`      -- Accuracy and F1 per dataset and model
- `attack_results.csv`        -- Explanation drift for every attack, model, dataset
- `defense_results.csv`       -- Post-defense drift comparison
- `fewshot_attack_results.csv`-- Few-shot model drift results

Plots are saved to `figures/`.

---

## Documentation

Extended documentation is available in the `docs/` directory:

- `docs/Architecture.md`            -- Full pipeline architecture
- `docs/AttackFramework.md`         -- Attack derivations and code
- `docs/DefenseArchitectures.md`    -- Loss formulations for all four defenses
- `docs/DriftMetric.md`             -- Formal definitions and variants
- `docs/FewShotAnalysis.md`         -- Prototypical network setup and results
- `docs/VivaQnA.md`                 -- Question and answer reference
- `docs/TechnicalReport.pdf`        -- Original 28-page report
- `docs/Presentation.pptx`          -- Project presentation deck

---

## Citation

If you use this framework in your research, please cite:

```
@misc{sagar2026xai,
  title  = {XAI Attack and Defense Framework with Few-Shot Learning},
  author = {Mithin Sagar S and Gokul Ram K and Kishore A G},
  year   = {2026},
  note   = {B.Tech AI and DS, VIT Chennai. Deep Learning DA-3.},
  url    = {https://github.com/mithinsagar/xai-attack-defense-framework}
}
```

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for
the full text. Copyright (c) 2026 Mithin Sagar S.
