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

## Data Alignment and Chunking

Run these commands from the repository root in Windows PowerShell. `.venv-a2` is recommended because it already contains `soundfile`, `scipy`, `numpy`, and `torch`.

Align dry/wet audio:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\align.py --dry data/raw/baseline_input.wav --wet data/raw/baseline_output.wav --out-dir data/aligned
```

Generate train/val/test chunks:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\chunk_audio.py --dry data/aligned/aligned_dry.wav --wet data/aligned/aligned_wet.wav --out-dir data/chunks --chunk-size 65536 --hop-size 65536 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Run the Dataset smoke test:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\dataset.py --metadata data/chunks/metadata.json --split train
```

Verify the full data pipeline:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\common\verify_data_pipeline.py
```

The current data processing module covers dry/wet sample-rate checks, cross-correlation delay estimation, automatic trimming, train/val/test chunk generation, and PyTorch Dataset loading.

By default, alignment treats peak amplitude `>= 1.0` as a warning instead of stopping the pipeline. Clipping risk and warning messages are written to `data/aligned/alignment_metadata.json`. To make clipping risk fatal, add `--strict-clipping`:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\align.py --dry data/raw/baseline_input.wav --wet data/raw/baseline_output.wav --out-dir data/aligned --strict-clipping
```

## A2 Failure Logs

The completed A2 smoke baseline status is preserved in `reports/STEP3_A2_COMPLETION_REPORT.md`. Future A2 run failures append entries under `## Failure Log` instead of overwriting the successful status. This keeps the historical success state separate from later troubleshooting records.

## Alignment Clipping Policy

- Default behavior: peak amplitude `>= 1.0` prints a warning and records clipping risk in metadata.
- Strict behavior: add `--strict-clipping` to fail alignment on clipping risk.
- Metadata fields include `dry_clipping_risk`, `wet_clipping_risk`, `strict_clipping`, and `warnings`.

## Inference

Run A1 inference with the A1 environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\inference\run_a1_inference.py --model outputs/a1_baseline/model.nam --input data/aligned/aligned_dry.wav --output outputs/a1_baseline/prediction.wav
```

Run A2 Lite and Full inference with the A2 environment:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\inference\run_a2_inference.py --model outputs/a2_baseline/model.nam --input data/aligned/aligned_dry.wav --lite-output outputs/a2_baseline/a2_lite_prediction.wav --full-output outputs/a2_baseline/a2_full_prediction.wav
```

All-in-one PowerShell entry:

```powershell
.\scripts\inference\run_all_inference.ps1 -ContinueOnError
```

Verify inference outputs:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\inference\verify_inference_outputs.py
```

Generate metrics comparison:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav
```

If the local NAM Python API cannot run offline inference, the scripts fail and write metadata explaining the reason. Do not copy input audio as prediction audio. If prediction audio is missing, comparison metrics remain `TBD`.

## Evaluation

`src/ntt/evaluation/metrics.py` provides MSE, MAE, ESR, Normalized MAE, SNR, and MRSTFT:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\metrics.py --pred data/aligned/aligned_dry.wav --target data/aligned/aligned_wet.wav
```

`src/ntt/evaluation/compare_models.py --artifact-only` generates the A1/A2 artifact comparison table:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py --artifact-only
```

When prediction audio is not available, ESR, MRSTFT, SNR, and other audio metrics are marked as `TBD`; do not treat artifact-only rows as audio-quality measurements.

## Notes

- A1 and A2 are safer in separate environments.
- A1 keeps `neural-amp-modeler==0.12.2` because that baseline has already been reproduced.
- A2 targets `neural-amp-modeler==0.13.0`.
- Do not commit large datasets, long audio files, checkpoint dumps, or bulk training outputs to GitHub.
- Keep only small proof artifacts, required configs, reports, and final small baseline models when appropriate.
