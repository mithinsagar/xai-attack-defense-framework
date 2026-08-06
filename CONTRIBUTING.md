# Contributing

**Author / Maintainer:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

Thanks for your interest in improving the XAI Attack and Defense Framework.

## Development Setup

```bash
git clone https://github.com/mithinsagar/xai-attack-defense-framework.git
cd xai-attack-defense-framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running Tests

```bash
pytest tests/
```

## Code Style

- Follow PEP 8.
- Use type hints for public functions.
- Add a header docstring to every new module in the style of the existing
  modules (project title, author line, description).

## Adding a New Attack

1. Create `attacks/<name>_attack.py` subclassing `attacks.attack_base.BaseAttack`.
2. Register it inside `attacks/attack_runner.py`.
3. Add a unit test under `tests/test_attacks.py`.
4. Document the attack in `docs/AttackFramework.md`.

## Adding a New Defense

1. Create `defenses/<name>_defense.py` with a `train_<name>` function.
2. Register it inside `defenses/defense_trainer.py`.
3. Add a unit test under `tests/test_defenses.py`.
4. Document the defense in `docs/DefenseArchitectures.md`.

## Reporting Issues

Please open an issue on GitHub with a minimal reproducer and the full
Python traceback.

## License

By contributing you agree that your contributions are licensed under the
MIT License (see `LICENSE`).
