# Attack Framework

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

## Threat Model

We consider a **white-box** attacker who has access to the model
parameters, the explanation method, and the input features. The goal is
to **maximally shift the explanation** while **minimally changing the
prediction**:

```
max_delta  ||E(x) - E(x + delta)||_1
subject to f(x) = f(x + delta),  ||delta||_inf <= epsilon
```

Where `E(.)` is the per-feature attribution function returned by SHAP /
LIME / Integrated Gradients.

## Attack Catalogue

| Attack | Name                                         | Target                     | Epsilon |
|-------:|----------------------------------------------|----------------------------|---------|
| 1      | Top-Feature Perturbation                     | Top-3 SHAP features        | 0.05    |
| 2      | Prediction-Stable Explanation Manipulation   | Bottom-3 SHAP features     | 0.10    |
| 3      | Gradient-Based Attack                        | Top-3 gradient features    | 0.05    |
| 4      | Targeted Explanation Attack                  | Top-5 SHAP/IG features     | 0.25    |

### Attack 1: Top-Feature Perturbation
Perturbs the most important features. Tests whether the model's
explanation is stable near the decision boundary of the features it
"cares about" most.

```python
perturbed = original.copy()
perturbed[21] += 0.05  # IsHTTPS
perturbed[22] += 0.05  # LineOfCode
perturbed[8]  += 0.05  # NoOfSubDomain
```

### Attack 2: Prediction-Stable Explanation Manipulation
Perturbs low-importance features. The most dangerous attack type
because it changes features the analyst would not expect to matter,
yet the explanation shifts significantly.

### Attack 3: Gradient-Based Attack
Uses the neural network's own gradients to identify the most sensitive
input direction. Produces the largest empirical explanation drift.

### Attack 4: Targeted Explanation Attack
Directly targets the top explanation features with larger perturbation
magnitudes. Represents the strongest single-shot attack.

## Attack Effectiveness Ranking

```
Attack 3 (Gradient)  >  Attack 4 (Targeted)  >  Attack 2 (Low)  >  Attack 1 (Top)
```

## Implementation Overview

Each attack subclasses `attacks.attack_base.BaseAttack` and implements a
single `perturb` method. `attacks.attack_runner.run_all_attacks()`
iterates every attack against every trained (dataset, model) combination
and writes a consolidated `results/attack_results.csv`.
