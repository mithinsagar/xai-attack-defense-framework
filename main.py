"""
=====================================================================
XAI Attack and Defense Framework - Pipeline Orchestrator
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Runs the end-to-end pipeline (or individual phases) via a single CLI
entrypoint. Each phase delegates to a dedicated module under `training/`,
`attacks/`, `defenses/`, and `explainers/`.

Usage
-----
    python main.py --phase all
    python main.py --phase preprocess
    python main.py --phase train_base
    python main.py --phase explain
    python main.py --phase attack
    python main.py --phase fewshot
    python main.py --phase defense
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict

import config
from utils.logger import get_logger
from utils.seed import set_seed


LOGGER = get_logger("main")


# ---------------------------------------------------------------------
# Phase entrypoints (lazy imports to keep startup fast)
# ---------------------------------------------------------------------
def _phase_preprocess() -> None:
    from data.preprocessor import preprocess_all_datasets
    LOGGER.info("Phase 1: Preprocessing all datasets")
    preprocess_all_datasets()


def _phase_train_base() -> None:
    from training.train_base_models import train_all_base_models
    LOGGER.info("Phase 2: Training all base models")
    train_all_base_models()


def _phase_explain() -> None:
    from explainers.explainer_utils import generate_all_explanations
    LOGGER.info("Phase 3: Generating explanations")
    generate_all_explanations()


def _phase_attack() -> None:
    from attacks.attack_runner import run_all_attacks
    LOGGER.info("Phase 4: Running attacks")
    run_all_attacks()


def _phase_fewshot() -> None:
    from training.train_fewshot import train_and_evaluate_fewshot
    LOGGER.info("Phase 5: Few-shot vulnerability analysis")
    train_and_evaluate_fewshot()


def _phase_defense() -> None:
    from training.train_defense_models import train_all_defenses
    LOGGER.info("Phase 6: Training defense models")
    train_all_defenses()


PHASES: Dict[str, Callable[[], None]] = {
    "preprocess": _phase_preprocess,
    "train_base": _phase_train_base,
    "explain":    _phase_explain,
    "attack":     _phase_attack,
    "fewshot":    _phase_fewshot,
    "defense":    _phase_defense,
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "XAI Attack and Defense Framework - Pipeline Orchestrator. "
            "Author: Mithin Sagar S "
            "(https://github.com/mithinsagar)."
        )
    )
    parser.add_argument(
        "--phase",
        choices=list(PHASES.keys()) + ["all"],
        default="all",
        help="Which pipeline phase to execute. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed used across NumPy, Python, PyTorch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    config.ensure_directories()
    set_seed(args.seed)
    LOGGER.info("Random seed set to %d", args.seed)

    if args.phase == "all":
        for name, phase_fn in PHASES.items():
            LOGGER.info("--- Running phase: %s ---", name)
            phase_fn()
    else:
        PHASES[args.phase]()

    LOGGER.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
