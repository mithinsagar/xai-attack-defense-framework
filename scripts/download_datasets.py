"""
=====================================================================
XAI Attack and Defense Framework - Dataset Download Helper
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

The three datasets used by this project require manual download because
they live behind portals (UCI, Kaggle, Mendeley) that either have their
own authentication flow or require accepting a licence. This helper
prints the exact URLs and expected file names.
"""

from __future__ import annotations

import config


def main() -> None:
    print("=" * 70)
    print("XAI Attack and Defense Framework - Dataset Download Helper")
    print("Author: Mithin Sagar S  (https://github.com/mithinsagar)")
    print("=" * 70)
    print("Place the downloaded CSV files under:")
    print(f"    {config.RAW_DATA_DIR}\n")

    for key, meta in config.DATASETS.items():
        print(f"[{key}]")
        print(f"  file : {meta['filename']}")
        print(f"  url  : {meta['source_url']}\n")

    print("Once downloaded, run:")
    print("    python main.py --phase preprocess")


if __name__ == "__main__":
    main()
