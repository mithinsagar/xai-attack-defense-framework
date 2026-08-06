# Data Directory

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

Raw dataset CSVs are **not** included in the repository because of size and
licensing. Download them yourself and place them in `data/raw/`.

## Required raw files

| File name expected in `data/raw/`               | Source URL |
|-------------------------------------------------|------------|
| `PhiUSIIL_Phishing_URL_Dataset.csv`             | <https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset> |
| `cybersecurity_intrusion_data.csv`              | <https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset> |
| `Fraudulent_online_shops_dataset.csv`           | <https://data.mendeley.com/datasets/m7xtkx7g5m/1> |

Detailed field-level information is in [`dataset_info.md`](dataset_info.md).

Once downloaded, run:

```bash
python main.py --phase preprocess
```

which will write processed arrays to `data/processed/` and fitted scalers
to `models/scalers/`.
