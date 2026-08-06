"""
=====================================================================
XAI Attack and Defense Framework - Package Setup
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="xai-attack-defense-framework",
    version="1.0.0",
    author="Mithin Sagar S",
    author_email="mithinsagar@example.com",
    url="https://github.com/mithinsagar/xai-attack-defense-framework",
    description=(
        "A framework for studying adversarial manipulation of explanations "
        "in cybersecurity ML models, with four attacks, four defenses, and "
        "few-shot vulnerability analysis."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=("tests", "notebooks", "docs")),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    entry_points={
        "console_scripts": [
            "xai-adf=main:main",
        ],
    },
)
