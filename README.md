# NeuralToneTransform

## Project Overview

NeuralToneTransform is a deep-learning audio black-box modeling project focused on learning tone transformations from paired dry and processed audio. The current milestone is to reproduce the official Neural Amp Modeler (NAM) baseline first, then extend the repository with in-house TCN, WaveNet, and LSTM model development.

## Current Status

- Step 1: Environment setup completed.
- Step 2: Official NAM baseline completed.
- Canonical NAM artifact available at `outputs/nam_baseline/model.nam`.

## Repository Structure

- `scripts/`
  Windows PowerShell entry points and Python utilities for environment setup, baseline preparation, training, finalization, and reproducibility verification.
- `configs/nam_baseline/`
  Generated NAM baseline configuration files: `data.json`, `model.json`, and `learning.json`.
- `data/raw/`
  Small baseline WAV files and smoke-test audio used for environment and NAM baseline validation.
- `outputs/nam_baseline/`
  Canonical baseline artifact (`model.nam`) plus official NAM training output folders.
- `reports/`
  Completion report, notes, and generated plots such as `waveform_smoke.png`.
- `logs/`
  Run logs for environment checks, baseline preparation, baseline training, and model finalization.

## Installation

Run the following commands from the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Check

Run the Python environment check directly:

```powershell
python scripts/check_env.py
```

This command prints status to the terminal and also writes `logs/step1_env_check.log`.

## Run Official NAM Baseline

Prepare baseline audio/configs and launch the official NAM baseline trainer:

```powershell
.\scripts\run_step2_nam_baseline.ps1
```

This script writes:

- `logs/step2_prepare_nam_baseline.log`
- `logs/step2_nam_training.log`

## Finalize Existing NAM Model

If you already have a trained NAM artifact and want to refresh the canonical archive path:

```powershell
.\scripts\finalize_step2_without_rerun.ps1
```

This script writes `logs/step2_finalize_model.log`.

## Expected Outputs

After a successful Step 1 + Step 2 baseline run, the repository should contain:

- `data/raw/baseline_input.wav`
- `data/raw/baseline_output.wav`
- `configs/nam_baseline/data.json`
- `configs/nam_baseline/model.json`
- `configs/nam_baseline/learning.json`
- `outputs/nam_baseline/model.nam`
- `reports/STEP1_STEP2_COMPLETION_REPORT.md`
- `logs/step1_env_check.log`
- `logs/step2_nam_training.log`

## Reproducibility Notes

- `scripts/prepare_nam_baseline.py` regenerates the NAM baseline configs for the current machine and keeps committed config paths free of personal absolute paths.
- `configs/nam_baseline/data.json` is expected to use repository-relative audio paths.
- Large user datasets, long audio captures, checkpoint dumps, and experimental weights should not be committed directly to GitHub.
- The current repository keeps only the small baseline proof artifacts required to show that the official NAM baseline has already run successfully.

## Reproduction Checklist

Use the built-in verification script after setup or after rerunning the baseline:

```powershell
.\scripts\verify_reproducibility.ps1
```

It checks the virtual environment, core imports, config hygiene, baseline WAV files, canonical model output, and required logs, then prints an overall PASS/FAIL summary.
