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

## GPU Environment

The project supports automatic CPU/GPU switching. CPU environments such as the existing `.venv` or `.venv-a2` are useful for smoke tests, reproducibility checks, and quick validation. Formal training should use a CUDA-enabled PyTorch environment.

Recommended global Conda environment:

```text
ntt-gpu-cu128
```

Install the environment from the repository root in Windows PowerShell:

```powershell
conda create -n ntt-gpu-cu128 python=3.11 -y
conda activate ntt-gpu-cu128
python -m pip install --upgrade pip setuptools wheel

Get-Content requirements.txt |
  Where-Object { $_ -notmatch "^(torch|torchaudio)==" } |
  Set-Content requirements-no-torch.txt

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
python -m pip install -r requirements-no-torch.txt
```

Install CUDA PyTorch before `requirements-no-torch.txt` so transitive `torch` dependencies are satisfied by the CUDA wheel instead of a CPU PyPI wheel.

Verify CUDA PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Run Formal Medium training with the global GPU Python:

```powershell
.\scripts\tcn\run_medium_formal_gpu.ps1
```

GatedTCN-Medium has completed 20-epoch formal GPU training. The formal checkpoint is available at `outputs/tcn_gated/medium/checkpoints/best.pt`.

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

## Custom Gated TCN

`Custom Gated TCN` is the self-developed model path for neural tone transformation. It uses causal dilated TCN blocks with gated tanh-sigmoid activations, residual connections, and skip aggregation to map dry audio to predicted wet audio. A1 and A2 remain reference baselines; the custom TCN is the project path for approaching their tone-modeling quality with our own architecture.

The TCN ablation has three variants:

- `GatedTCN-Small`: 16 residual/skip channels, 12 layers, 2053-sample receptive field.
- `GatedTCN-Medium`: 32 residual/skip channels, 20 layers, 4093-sample receptive field.
- `GatedTCN-Large`: 48 residual/skip channels, 22 layers, 8189-sample receptive field.

Large is designed to exceed the A2 reference receptive field of roughly 6350 samples while still supporting CPU smoke runs.

Train the medium model directly:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\tcn\train.py --model-config configs/tcn_gated/medium.json --training-config configs/tcn_gated/training.json
```

Run inference:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\tcn\infer.py --checkpoint outputs/tcn_gated/medium/checkpoints/best.pt --input data/aligned/aligned_dry.wav --output outputs/tcn_gated/medium/prediction.wav
```

Verify TCN outputs:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\tcn\verify_tcn.py --output-dir outputs/tcn_gated/medium
```

Compare with A1/A2:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav `
  --custom-tcn-pred outputs/tcn_gated/medium/prediction.wav `
  --custom-tcn-name GatedTCN-Medium `
  --custom-tcn-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt
```

The training and inference scripts use `device=auto` by default. They use CUDA automatically when `torch.cuda.is_available()` is true, otherwise they fall back to CPU. CUDA is supported but not required.

Run smoke ablation for Small/Medium/Large:

```powershell
.\scripts\tcn\run_tcn_ablation.ps1 -Smoke -ContinueOnError
```

Run formal 20-epoch ablation:

```powershell
.\scripts\tcn\run_tcn_ablation.ps1 -ContinueOnError
```

Run Medium formal 20-epoch training only:

```powershell
.\scripts\tcn\run_tcn_ablation.ps1 -SkipSmall -SkipLarge -ContinueOnError
```

The formal configs keep `num_epochs=20`. With `device=auto`, CUDA is used automatically when available; CPU-only runtimes can be very slow and should keep formal status as pending until the run actually completes.

GatedTCN-Medium has completed 20-epoch formal GPU training. The formal checkpoint is available at `outputs/tcn_gated/medium/checkpoints/best.pt`.

## Final Custom TCN Model

The final selected custom model is GatedTCN-Medium.

- Checkpoint: `outputs/tcn_gated/medium/checkpoints/best.pt`
- Prediction: `outputs/tcn_gated/medium/prediction.wav`
- Model card: `reports/FINAL_TCN_MODEL_CARD.md`
- Final experiment summary: `reports/FINAL_EXPERIMENT_SUMMARY.md`
- Subjective listening template: `reports/SUBJECTIVE_LISTENING_TEMPLATE.md`
- Multi-seed experiment plan: `reports/MULTI_SEED_EXPERIMENT_PLAN.md`
- External dataset evaluation plan: `reports/EXTERNAL_DATASET_EVALUATION_PLAN.md`

Small and Large have completed formal 20-epoch GPU ablation runs. They are not used as the final selected model. Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.

Reproduce the Medium formal GPU run:

```powershell
.\scripts\tcn\run_medium_formal_gpu.ps1
```

Formal configs:

- `configs/tcn_gated/training_formal_small.json`
- `configs/tcn_gated/training_formal_medium.json`
- `configs/tcn_gated/training_formal_large.json`

Benchmark inference speed:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\tcn\benchmark.py --checkpoint outputs/tcn_gated/medium/checkpoints/best.pt --input data/aligned/aligned_dry.wav --device auto
```

Generate waveform and spectrogram figures:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\plot_comparison.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav `
  --tcn-small-pred outputs/tcn_gated/small/prediction.wav `
  --tcn-medium-pred outputs/tcn_gated/medium/prediction.wav `
  --tcn-large-pred outputs/tcn_gated/large/prediction.wav `
  --out-dir reports/figures
```

Generate the full comparison table:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav `
  --tcn-small-pred outputs/tcn_gated/small/prediction.wav `
  --tcn-small-checkpoint outputs/tcn_gated/small/checkpoints/best.pt `
  --tcn-medium-pred outputs/tcn_gated/medium/prediction.wav `
  --tcn-medium-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt `
  --tcn-large-pred outputs/tcn_gated/large/prediction.wav `
  --tcn-large-checkpoint outputs/tcn_gated/large/checkpoints/best.pt
```

Held-out test evaluation for TCN Medium:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --tcn-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt `
  --model-name GatedTCN-Medium `
  --out reports/test_metrics_tcn_medium.json `
  --device auto
```

Held-out test evaluation for A1/A2 predictions:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a1_baseline/prediction.wav `
  --model-name A1-baseline `
  --out reports/test_metrics_a1.json

.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a2_baseline/a2_lite_prediction.wav `
  --model-name A2-Lite `
  --out reports/test_metrics_a2_lite.json

.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a2_baseline/a2_full_prediction.wav `
  --model-name A2-Full `
  --out reports/test_metrics_a2_full.json
```

Benchmark A1/A2 NAM inference RTF:

```powershell
.\.venv\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a1_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a1 `
  --out outputs/a1_baseline/benchmark.json

.\.venv-a2\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a2_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a2-lite `
  --out outputs/a2_baseline/a2_lite_benchmark.json

.\.venv-a2\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a2_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a2-full `
  --out outputs/a2_baseline/a2_full_benchmark.json
```

Final comparison should use held-out test metrics plus RTF. Full aligned-audio metrics are useful diagnostics, but final experiment claims should prioritize `reports/test_metrics_*.json` and benchmark JSON outputs.

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
