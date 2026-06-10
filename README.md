# NeuralToneTransform

## Project Overview

NeuralToneTransform is a neural black-box modeling project for guitar and bass amplifier, cabinet, and effects-chain tone transformations. The current reproduction route is:

1. reproduce A1 / legacy NAM WaveNet baseline;
2. reproduce A2 / PackedWaveNet baseline;
3. build custom lightweight models and compare experiments.

## Roadmap

- Step 1: Environment setup
- Step 2: A1 / legacy NAM baseline
- Step 3: A2 / PackedWaveNet baseline
- Step 4: Data alignment and chunking
- Step 5: Custom lightweight model
- Step 6: Evaluation and comparison

## Repository Structure

- `configs/a1_baseline/`
  A1 legacy NAM config files: `data.json`, `model.json`, and `learning.json`.
- `configs/a2_baseline/`
  A2 config files: `data.json`, `model_packed.json`, and `learning.json`.
- `outputs/a1_baseline/`
  Canonical A1 artifact at `model.nam` plus A1 run outputs.
- `outputs/a2_baseline/`
  Canonical A2 artifact at `model.nam`, extracted Lite/Full directories, and A2 run outputs.
- `scripts/a1/`
  Windows PowerShell and Python entry points for preparing, training, finalizing, and verifying A1.
- `scripts/a2/`
  Windows PowerShell and Python entry points for preparing, training, inspecting, finalizing, and verifying A2.
- `scripts/common/`
  Shared dependency, audio-pair, and reproducibility verification helpers.
- `src/ntt/`
  Project utility package for data preparation, evaluation, and path/audio helpers.

The original `configs/nam_baseline/`, `outputs/nam_baseline/`, and Step 2 scripts are preserved as legacy proof artifacts. New work should use the A1/A2 directories.

## How to Run A1

Run from the repository root in Windows PowerShell:

```powershell
.\scripts\a1\run_a1_baseline.ps1
.\scripts\a1\verify_a1_baseline.ps1
```

To reuse the already trained baseline model without rerunning training:

```powershell
.\scripts\a1\run_a1_baseline.ps1 -SkipTrain
```

## How to Run A2

Run from the repository root in Windows PowerShell:

```powershell
.\scripts\a2\run_a2_baseline.ps1
.\scripts\a2\verify_a2_baseline.ps1
```

To create `.venv-a2`, install A2 dependencies, and generate configs without training:

```powershell
.\scripts\a2\run_a2_baseline.ps1 -SkipTrain
```

## Expected Outputs

A1 output:

- `outputs/a1_baseline/model.nam`

A2 output:

- `outputs/a2_baseline/model.nam`
- `reports/a2_model_inspection.md`, showing whether `SlimmableContainer` appears in the exported `.nam`.

## Notes

- A1 and A2 are safer in separate environments.
- A1 keeps `neural-amp-modeler==0.12.2` because that baseline has already been reproduced.
- A2 targets `neural-amp-modeler==0.13.0`.
- Do not commit large datasets, long audio files, checkpoint dumps, or bulk training outputs to GitHub.
- Keep only small proof artifacts, required configs, reports, and final small baseline models when appropriate.
