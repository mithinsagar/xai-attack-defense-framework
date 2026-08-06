# Viva Questions and Answers

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

Compiled from the project defence for Deep Learning (DA-3), VIT Chennai.

---

## Fundamental Concepts

**Q1. Why do we need Explainable AI? Cannot we just trust the model's
accuracy?**
High accuracy does not guarantee the model is learning the right
features. A model could reach 99 percent accuracy by exploiting spurious
correlations (for example a phishing detector that relies on URL length
rather than actual malicious indicators). XAI reveals what the model
learned, enabling debugging, regulatory compliance (GDPR Article 22
mandates explanation of automated decisions), and trust calibration. In
security-critical domains, a wrong explanation can be more dangerous
than a wrong prediction.

**Q2. What is the difference between SHAP and LIME? When would you
prefer one over the other?**
SHAP is grounded in game theory (Shapley values) and provides globally
consistent feature attributions with theoretical guarantees (local
accuracy, missingness, consistency). LIME fits a local linear model
around a single prediction using perturbation-based sampling. SHAP is
preferred when: exact computation is possible (TreeSHAP for tree
models), global aggregation of explanations is needed, or theoretical
guarantees matter. LIME is preferred when the model is extremely large
and SHAP's KernelExplainer is too slow, or when a simple intuitive
explanation is sufficient.

**Q3. Why did you use Integrated Gradients instead of vanilla
gradients?**
Vanilla gradients suffer from gradient saturation (for sigmoid or ReLU
activations, gradients can be near zero even for highly important
features) and violate the sensitivity axiom (if changing a feature
changes the prediction, vanilla gradients might still assign zero
attribution). Integrated Gradients solves both problems by integrating
along a path from a baseline, satisfying both sensitivity and
implementation invariance.

**Q4. What is the intuition behind Shapley values?**
Each feature is a player in a cooperative game and the model prediction
is the payout. The Shapley value of a feature is its average marginal
contribution across all possible subsets. It is the only attribution
method satisfying efficiency, symmetry, linearity, and null player
properties simultaneously.

---

## Model Design

**Q5. Why Logistic Regression, XGBoost, and MLP? Why not Random Forest
or SVM?**
These three models span the spectrum of complexity: linear (LR),
ensemble tree (XGBoost), deep learning (MLP). Random Forest is
functionally similar to XGBoost for SHAP analysis but slower. SVM's
KernelSHAP is intractable on 235K samples. CNN and RNN are designed for
spatial or sequential data, not tabular. Transformer is overkill for
tabular binary classification with fewer than 60 features.

**Q6. Why do you use BatchNorm and Dropout only in the base MLP but not
in `MLPSmall`?**
`MLPSmall` is used for the IDS dataset (~9.5K samples). BatchNorm and
Dropout can destabilise training on small datasets, so we deliberately
remove them.

---

## Attacks

**Q7. Which attack is strongest, and why?**
Attack 3 (Gradient-Based) consistently produces the highest drift
because it uses the model's own gradients to identify the most
sensitive perturbation direction.

**Q8. Attack 2 preserves the prediction. Why is it dangerous?**
It shifts the explanation via features the analyst would not expect to
matter. The prediction looks unchanged (so no alarm is raised), but
the reasoning shown to the analyst is wrong.

---

## Defenses

**Q9. Why does the Hybrid Defense outperform every individual defense?**
It combines complementary loss terms: stability (output smoothness),
adversarial (prediction robustness), and smoothness (consistency across
perturbation magnitudes). No single term covers all three properties.

**Q10. Why is Stability Training alone almost as good as the Hybrid?**
Because output smoothness transitively enforces explanation smoothness:
if the outputs are similar under small perturbations, the gradients
(and therefore Integrated Gradients) are also similar.

---

## Few-Shot Learning

**Q11. Why are few-shot models more vulnerable?**
Limited data leads to a less well-defined decision boundary, larger and
more erratic gradients, and a less smooth explanation landscape. All
three amplify explanation drift under adversarial perturbation.

---

## Advanced

**Q12. What could you do to make explanations provably robust?**
Randomised smoothing can give certified robustness bounds on
explanations. Ensembling multiple explanation methods (SHAP + LIME + IG)
also shrinks the manipulation surface. Input denoising via an
autoencoder is another promising direction.
