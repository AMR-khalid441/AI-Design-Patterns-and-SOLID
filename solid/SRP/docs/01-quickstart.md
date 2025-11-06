# Quickstart

This guide shows how to run the SRP examples from the repo root. Commands are provided for Linux/macOS (bash) and Windows (PowerShell).

## Violation (god class)

Recommended isolated environment:

```bash
conda create -n srp-violation python=3.11 -y
conda activate srp-violation
python -m pip install -r solid/SRP/violation/requirements.txt
```

Run (Linux/macOS):

```bash
python -m solid.SRP.violation.run_training --config solid/SRP/shared/fixtures/model_config.yaml
```

Run (Windows PowerShell):

```powershell
python -m solid.SRP.violation.run_training --config solid\SRP\shared\fixtures\model_config.yaml
```

Optional overrides (CLI > YAML > defaults):

```bash
python -m solid.SRP.violation.run_training --epochs 1 --lr 0.0005 --batch_size 32 --seed 123
```

## Solution (SRP-applied components)

Recommended isolated environment:

```bash
conda create -n srp-solution python=3.11 -y
conda activate srp-solution
python -m pip install -r solid/SRP/solution/requirements.txt
```

Run (Linux/macOS):

```bash
python -m solid.SRP.solution.run_training --config solid/SRP/shared/fixtures/model_config.yaml
```

Run (Windows PowerShell):

```powershell
python -m solid.SRP.solution.run_training --config solid\SRP\shared\fixtures\model_config.yaml
```

## Notes
- Always run from the repository root to keep paths correct.
- Config precedence: CLI flags override YAML, which overrides built-in defaults.
- See also: `docs/02-architecture.md` for architecture rationale and layer responsibilities.
 - On very small CSVs, the split may automatically fall back to non-stratified to avoid errors. You can force non-stratified with `--no_stratify`.


