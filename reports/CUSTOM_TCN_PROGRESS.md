# Custom Gated TCN Progress

## Current Status

The Custom Gated TCN code path is implemented under `src/ntt/tcn/`. Small, Medium, and Large variants completed CPU smoke ablation runs with training, checkpointing, full-file inference, verification, benchmark output, figure generation, full-file comparison, and held-out test split evaluation.

Formal training support is implemented through `configs/tcn_gated/training_formal_*.json` and `scripts/tcn/run_tcn_ablation.ps1`.

Medium 20-epoch formal training pending due to CPU-only runtime.

## Formal Training

Medium formal config:

- config: `configs/tcn_gated/training_formal_medium.json`
- epochs: 20
- batch size: 4
- device: `auto`
- max train batches: null
- max val batches: null

Formal command:

```powershell
.\scripts\tcn\run_tcn_ablation.ps1 -SkipSmall -SkipLarge -ContinueOnError
```

Current runtime check:

- `torch`: 2.11.0+cpu
- CUDA available: false
- status: Formal 20-epoch training pending due to CPU-only runtime.

## Model Configurations

| Model | Channels | Layers | Receptive Field | Parameters | Config |
| --- | ---: | ---: | ---: | ---: | --- |
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | `configs/tcn_gated/small.json` |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | `configs/tcn_gated/medium.json` |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | `configs/tcn_gated/large.json` |

Large exceeds the A2 reference receptive field of roughly 6350 samples while remaining small enough for CPU smoke verification.

## Smoke Training Status

| Model | Device | Epochs | Train Batches | Val Batches | Train Loss | Val Loss | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GatedTCN-Small | cpu | 1 | 1 | 1 | 0.198326 | 0.163366 | PASS |
| GatedTCN-Medium | cpu | 1 | 1 | 1 | 0.123419 | 0.858192 | PASS |
| GatedTCN-Large | cpu | 1 | 1 | 1 | 0.139288 | 0.103801 | PASS |

## Held-out Test Evaluation

Held-out test split evaluation support is implemented in `src/ntt/evaluation/evaluate_test_split.py`.

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 |
| GatedTCN-Medium | 0.0294477 | 0.144289 | 2.63712 | 5.13135 | -3.5008 |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 |

These TCN values are from CPU smoke checkpoints, not formal 20-epoch checkpoints.

## Benchmark Status

TCN benchmark support is implemented in `src/ntt/tcn/benchmark.py`. A1/A2 NAM benchmark support is implemented in `scripts/inference/benchmark_nam_inference.py`.

| Model | Device / Env | RTF | Samples/s | Avg Chunk Latency ms |
| --- | --- | ---: | ---: | ---: |
| A1 baseline | `.venv` | 0.0670541 | 715840.39 | 91.00 |
| A2-Lite baseline | `.venv-a2` | 0.0517085 | 928280.72 | 70.18 |
| A2-Full baseline | `.venv-a2` | 0.0708378 | 677604.35 | 96.14 |
| GatedTCN-Small | cpu | 0.0738286 | 650154.30 | 400.72 |
| GatedTCN-Medium | cpu | 0.178694 | 268615.21 | 970.00 |
| GatedTCN-Large | cpu | 0.336964 | 142448.60 | 1829.17 |

## Figure Generation Status

Generated under `reports/figures/`:

- `waveform_overlay.png`
- `error_waveform.png`
- `spectrogram_target.png`
- `spectrogram_a1.png`
- `spectrogram_a2_lite.png`
- `spectrogram_a2_full.png`
- `spectrogram_tcn_small.png`
- `spectrogram_tcn_medium.png`
- `spectrogram_tcn_large.png`
- `spectrogram_error_tcn_medium.png`

Figure notes are in `reports/FIGURE_ANALYSIS.md`.

## Comparison with A1/A2

`src/ntt/evaluation/compare_models.py` now writes rows for A1 baseline, A2-Lite baseline, A2-Full baseline, GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large. The table includes full-file metrics, held-out test metrics, and RTF.

## Remaining Work

- run Medium formal 20-epoch training on a CUDA-capable runtime
- optionally run Small and Large formal 20-epoch ablations
- repeat with multiple seeds
- compare against A1/A2 using formal TCN checkpoints
- add subjective listening notes after formal training
