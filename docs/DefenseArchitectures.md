# Defense Architectures

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

All four defenses use one of two architectures defined in
`models/robust_mlp.py`:

- `RobustMLP`     : 128 -> 64 -> 2 (used by Defense 1)
- `RobustMLP_v2`  : 256 -> 128 -> 64 -> 2 with BatchNorm + Dropout
                    (used by Defenses 2, 3, 4)

All defenses output **2 logits** and use `CrossEntropyLoss`, which is
required for targeted Integrated Gradients attribution.

## Defense 1: Stability Training

```
L = CE(f(x), y) + lambda_1 * mean|f(x) - f(x + N(0, sigma^2))|
```

- sigma = 0.01
- lambda_1 = 0.1
- Epochs = 50

Forces the model to produce similar outputs for clean and slightly
noisy inputs. Empirically the most impactful single-mechanism defense.

## Defense 2: Adversarial Training

```
L = CE(f(x), y) + lambda_2 * CE(f(x + delta_adv), y)
```

- delta_adv ~ N(0, 0.05^2)
- lambda_2 = 0.5
- Epochs = 60

Improves prediction robustness under perturbation, but less effective at
explanation robustness than stability training.

## Defense 3: Explanation Regularization

```
L = CE(f(x), y) + lambda_3 * mean|IG(x) - IG(x + N(0, sigma^2))|
```

- sigma = 0.02
- lambda_3 = 0.2
- Epochs = 40

Directly regularises the explanation itself. IG is expensive, so only
200 samples per epoch participate in the explanation loss.

## Defense 4: Hybrid Defense  (BEST)

```
L_hybrid = CE(f(x), y)
         + w_stability * mean|f(x) - f(x + N(0, sigma_s^2))|
         + w_adv       * CE(f(x + N(0, sigma_l^2)), y)
         + w_smooth    * mean|f(x + N(0, sigma_s^2)) - f(x + N(0, sigma_l^2))|
```

- sigma_s = 0.01, sigma_l = 0.05
- w_stability = w_adv = 0.2, w_smooth = 0.1
- Epochs = 50

Achieves ~91.1 percent explanation drift reduction, the best of all
four defenses. Saved as `models/defense_models/best_defense_model.pt`.

## Defense Effectiveness Ranking

```
Defense 4 (Hybrid)  >  Defense 1 (Stability)  >  Defense 2 (Adversarial)  >  Defense 3 (Explanation Reg.)
```
