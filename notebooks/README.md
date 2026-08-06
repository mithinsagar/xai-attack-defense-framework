# Notebooks

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

These are the original Jupyter notebooks that produced every result in
the technical report and presentation. They are preserved verbatim for
reproducibility and correspond one-to-one with the six pipeline phases:

| Notebook                                      | Phase | Contents |
|-----------------------------------------------|-------|----------|
| `01_Project_Setup.ipynb`                      | 1     | Directory setup, dataset loading, preprocessing |
| `02_Base_Model_Training.ipynb`                | 2     | Logistic, XGBoost, MLP training and evaluation |
| `03_Explanation_Generation.ipynb`             | 3     | SHAP, LIME, IG, Permutation Importance |
| `04_Attack_Generation.ipynb`                  | 4     | All four attacks against all trained models |
| `05_FewShot_Vulnerability.ipynb`              | 5     | Prototypical Network few-shot attacks |
| `06_Defense_Model.ipynb`                      | 6     | Four defense architectures + best model save |

For programmatic execution, use the packaged modules under `../` and
run `python main.py --phase all` from the project root.
