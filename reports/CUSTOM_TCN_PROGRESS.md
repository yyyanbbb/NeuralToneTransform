# Custom Gated TCN Progress

## Current Status

GatedTCN-Medium has completed 20-epoch formal training on GPU. The prediction audio, held-out test metrics, inference benchmark, and comparison table have been regenerated from the formal checkpoint.

Small and Large remain CPU smoke checkpoints. They should continue to be labeled as smoke results until their own 20-epoch formal runs complete.

## Formal Medium Training

| Field | Value |
|---|---|
| Status | Completed |
| Device | cuda |
| CUDA Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Epochs | 20 |
| Completed Epochs | 20 |
| Best Epoch | 19 |
| Best Val Loss | 0.00628652 |
| Parameter Count | 167553 |
| Receptive Field | 4093 |
| Training Time Seconds | 125.642443 |
| Model Config | `configs/tcn_gated/medium.json` |
| Training Config | `configs/tcn_gated/training_formal_medium.json` |
| Checkpoint | `outputs/tcn_gated/medium/checkpoints/best.pt` |

## Model Configurations

| Model | Channels | Layers | Receptive Field | Parameters | Config |
| --- | ---: | ---: | ---: | ---: | --- |
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | `configs/tcn_gated/small.json` |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | `configs/tcn_gated/medium.json` |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | `configs/tcn_gated/large.json` |

Large exceeds the A2 reference receptive field of roughly 6350 samples while remaining small enough for CPU smoke verification.

## Training Status

| Model | Device | Epochs | Training Config | Status |
| --- | --- | ---: | --- | --- |
| GatedTCN-Small | cpu | 1 | `configs/tcn_gated/training_smoke_small.json` | Smoke checkpoint, not formal training |
| GatedTCN-Medium | cuda | 20 | `configs/tcn_gated/training_formal_medium.json` | Formal 20-epoch GPU checkpoint |
| GatedTCN-Large | cpu | 1 | `configs/tcn_gated/training_smoke_large.json` | Smoke checkpoint, not formal training |

## Held-out Test Evaluation

Held-out test split evaluation support is implemented in `src/ntt/evaluation/evaluate_test_split.py`.

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 |

GatedTCN-Medium test metrics are based on the 20-epoch formal GPU checkpoint. Small and Large are still CPU smoke checkpoint metrics.

## Benchmark Status

TCN benchmark support is implemented in `src/ntt/tcn/benchmark.py`. A1/A2 NAM benchmark support is implemented in `scripts/inference/benchmark_nam_inference.py`.

| Model | Device / Env | RTF | Samples/s | Avg Chunk Latency ms |
| --- | --- | ---: | ---: | ---: |
| A1 baseline | `.venv` | 0.0670541 | 715840.39 | 91.00 |
| A2-Lite baseline | `.venv-a2` | 0.0517085 | 928280.72 | 70.18 |
| A2-Full baseline | `.venv-a2` | 0.0708378 | 677604.35 | 96.14 |
| GatedTCN-Small | cpu | 0.0738286 | 650154.30 | 400.72 |
| GatedTCN-Medium | cuda | 0.0132906 | 3611571.84 | 71.71 |
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

`src/ntt/evaluation/compare_models.py` writes rows for A1 baseline, A2-Lite baseline, A2-Full baseline, GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large. The table includes full-file metrics, held-out test metrics, and RTF.

`reports/experiment_comparison.md` has been regenerated from the formal Medium prediction, `reports/test_metrics_tcn_medium.json`, and `outputs/tcn_gated/medium/benchmark.json`.

## Remaining Work

- optionally run Small and Large formal 20-epoch ablations
- repeat with multiple seeds
- add subjective listening notes after formal training
