"""
=====================================================================
XAI Attack and Defense Framework - Summary Report Generator
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Aggregates all CSVs under `results/` into a single Markdown summary
report at `results/summary_report.md`. Useful as a quick sanity-check
after a fresh pipeline run.
"""

from __future__ import annotations

import pandas as pd

import config


def _try_read(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def main() -> None:
    lines = [
        "# XAI Attack and Defense Framework - Summary Report",
        "",
        "**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))",
        "",
        "---",
        "",
    ]

    for name, header in (
        ("baseline_results.csv",        "## Baseline Model Performance"),
        ("attack_results.csv",          "## Attack Results (Explanation Drift)"),
        ("defense_results.csv",         "## Defense Results (Explanation Drift)"),
        ("fewshot_attack_results.csv",  "## Few-Shot Attack Results"),
    ):
        df = _try_read(config.RESULTS_DIR / name)
        lines.append(header)
        lines.append("")
        if df is None or df.empty:
            lines.append("_No data available yet - run the corresponding phase first._")
        else:
            lines.append(df.to_markdown(index=False))
        lines.append("")
        lines.append("---")
        lines.append("")

    out = config.RESULTS_DIR / "summary_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to: {out}")


if __name__ == "__main__":
    main()
