"""
=====================================================================
XAI Attack and Defense Framework - Shared Constants
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

# Author / project metadata
AUTHOR = "Mithin Sagar S"
GITHUB_URL = "https://github.com/mithinsagar"
PROJECT_NAME = "XAI Attack and Defense Framework"

# Dataset keys used throughout the codebase
DATASET_KEYS = ("phishing", "ids", "fraud")

# Model keys used throughout the codebase
MODEL_KEYS = ("logistic", "xgboost", "mlp")

# Attack keys
ATTACK_KEYS = (
    "attack1_top_feature",
    "attack2_prediction_stable",
    "attack3_gradient",
    "attack4_targeted",
)

# Defense keys
DEFENSE_KEYS = (
    "defense1_stability",
    "defense2_adversarial",
    "defense3_explanation_reg",
    "defense4_hybrid",
)

# Plot styling constants
MODERN_COLORS = ("#4C72B0", "#DD8452")
SEABORN_STYLE = "whitegrid"
SEABORN_CONTEXT = "talk"

# Numerical defaults
DEFAULT_EPS = 1e-9
