# STEP1 STEP2 A1 COMPLETION REPORT

## Status

- Step 1 environment setup: completed from the existing project run.
- Step 2 A1 / legacy NAM WaveNet baseline: completed and preserved.
- Canonical A1 model: `outputs/a1_baseline/model.nam`.

## Environment

- OS workflow: Windows PowerShell.
- A1 virtual environment: `.venv`.
- Python observed in `.venv`: `3.13.9`.
- NAM package: `neural-amp-modeler==0.12.2`.
- CUDA: previous baseline report recorded CPU execution.

## Dependency Versions

A1 versions are pinned in `requirements-a1.txt`:

- `torch==2.11.0`
- `torchaudio==2.11.0`
- `librosa==0.11.0`
- `matplotlib==3.10.8`
- `tensorboard==2.20.0`
- `soundfile==0.13.1`
- `scipy==1.17.1`
- `numpy==2.4.4`
- `pandas==3.0.2`
- `neural-amp-modeler==0.12.2`

## Data Files

- Input WAV: `data/raw/baseline_input.wav`
- Output WAV: `data/raw/baseline_output.wav`

## Config Files

- `configs/a1_baseline/data.json`
- `configs/a1_baseline/model.json`
- `configs/a1_baseline/learning.json`

The config paths use repository-relative audio paths.

## Training Command

```powershell
.\scripts\a1\run_a1_baseline.ps1
```

For validation without retraining:

```powershell
.\scripts\a1\run_a1_baseline.ps1 -SkipTrain
```

## Output Model

- Legacy source path kept: `outputs/nam_baseline/model.nam`
- A1 canonical path: `outputs/a1_baseline/model.nam`
- Model size observed before refactor: `297454` bytes

## Acceptance Result

A1 has been standardized into the new A1 directory layout without deleting the old NAM baseline.

```powershell
.\scripts\a1\verify_a1_baseline.ps1
```

Latest PowerShell verification result:

```text
pass_count: 12
fail_count: 0
OVERALL: PASS
```

Latest shared Python verification result:

```text
pass_count: 15
fail_count: 0
OVERALL: PASS
```
