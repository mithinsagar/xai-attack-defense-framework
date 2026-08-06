# Few-Shot Vulnerability Analysis

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

## Setup

- 50 samples per class from the Phishing dataset (100 total).
- Prototypical Network with a 2-layer encoder (128 -> 64).
- N-way = 2, K-shot = 5, Q-query = 10, Episodes = 300, Adam lr = 0.001.

## Architecture

```python
Encoder = Linear(d, 128) -> ReLU -> Linear(128, 64)

Classifier logits = -||encoder(query) - class_prototype||^2
```

Class prototypes are the mean embeddings of the support set.

## Attack Results

| Attack             | Full Model Drift | Few-Shot Drift |
|--------------------|------------------|----------------|
| Attack 1 (Top)     | 0.0230           | 0.0412         |
| Attack 2 (Low)     | 0.0185           | 0.0356         |
| Attack 3 (Gradient)| 0.0387           | 0.0623         |
| Attack 4 (Targeted)| 0.0251           | 0.0498         |

Few-shot models exhibit **60 to 80 percent higher explanation drift**
across all attack types because:

1. The decision boundary is less well-defined with limited data.
2. Gradient magnitudes are larger and more erratic.
3. The explanation landscape is less smooth than in well-trained models.
