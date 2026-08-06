# Dataset Information

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

This project uses three publicly available cybersecurity datasets spanning
distinct domains (phishing URL classification, network intrusion detection,
and online-shop fraud detection).

---

## 1. PhiUSIIL Phishing URL Dataset

- **Source:** UCI Machine Learning Repository
- **Link:** <https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- **Samples:** 235,795
- **Features:** 56 raw, 51 after preprocessing (5 dropped as leakage or IDs)
- **Target column:** `label` (0 = Legitimate, 1 = Phishing)
- **Dropped columns:** `FILENAME`, `URL`, `Domain`, `Title`, `URLSimilarityIndex`
- **File name expected in `data/raw/`:** `PhiUSIIL_Phishing_URL_Dataset.csv`

Feature categories: URL-based (URLLength, DomainLength, TLDLength, NoOfSubDomain,
HasObfuscation, NoOfDigitsInURL), HTML-based (LineOfCode, LargestLineLength,
NoOfImage, NoOfCSS, NoOfJS), and Security (IsHTTPS, HasSocialNet,
HasHiddenFields, HasPasswordField).

Feature engineering note: `URLSimilarityIndex` is removed because it exhibits
near-constant behavior for the phishing class, which would trivialise
classification and inflate model reliance on a single feature.

---

## 2. Cybersecurity Intrusion Detection Dataset

- **Source:** Kaggle (dnkumars)
- **Link:** <https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset>
- **Samples:** 9,537
- **Features:** 11 raw, 10 after dropping `session_id`
- **Target column:** `attack_detected` (0 = Normal, 1 = Attack)
- **Dropped columns:** `session_id`
- **File name expected in `data/raw/`:** `cybersecurity_intrusion_data.csv`

Feature categories: network packet size, protocol type, login attempts,
failed logins, session duration, encryption used, IP reputation score.

---

## 3. Fraudulent Online Shops Dataset

- **Source:** Mendeley Data
- **Link:** <https://data.mendeley.com/datasets/m7xtkx7g5m/1>
- **Samples:** 1,140
- **Features:** 26 raw, 24 after preprocessing
- **Target column:** `Label` (0 = Legitimate, 1 = Fraud)
- **Dropped columns:** `Online shop URL`, `Domain registration date`
- **File name expected in `data/raw/`:** `Fraudulent_online_shops_dataset.csv`

Feature categories: domain features (domain length, top domain length),
security features (SSL issuer, Trustpilot score), payment features (credit
card presence, crypto payment).

---

## Preprocessing pipeline (shared)

Implemented in `data/preprocessor.py`:

1. Drop non-informative columns.
2. Factorise categorical columns.
3. Mean-impute missing values.
4. `StandardScaler` normalisation.
5. Stratified 80/20 train-test split with `random_state=42`.

Processed artefacts are written to `data/processed/`:

```
X_train_<key>.npy       X_test_<key>.npy
y_train_<key>.npy       y_test_<key>.npy
feature_<key>.pkl       (feature name list)
```

Fitted scalers are persisted to `models/scalers/scaler_<key>.pkl`.
