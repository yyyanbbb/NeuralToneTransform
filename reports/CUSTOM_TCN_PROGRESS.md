# Custom Gated TCN Progress

## Current Status

The Custom Gated TCN code path is implemented under `src/ntt/tcn/`. Small, Medium, and Large variants completed CPU smoke ablation runs with training, checkpointing, full-file inference, verification, benchmark output, and comparison-table integration.

Formal 20-epoch training pending due to CPU-only runtime.

## Formal Experiment Plan

- Train `GatedTCN-Small`, `GatedTCN-Medium`, and `GatedTCN-Large` for 20 epochs using the formal configs in `configs/tcn_gated/`.
- Run full-file inference for each variant against `data/aligned/aligned_dry.wav`.
- Verify outputs with `src/ntt/tcn/verify_tcn.py`.
- Benchmark each checkpoint for RTF, throughput, and chunk latency.
- Compare A1, A2-Lite, A2-Full, and all TCN variants in `reports/experiment_comparison.md`.
- Generate waveform and spectrogram figures under `reports/figures/`.

## Model Configurations

| Model | Channels | Layers | Receptive Field | Parameters | Config |
| --- | ---: | ---: | ---: | ---: | --- |
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | `configs/tcn_gated/small.json` |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | `configs/tcn_gated/medium.json` |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | `configs/tcn_gated/large.json` |

Large is designed to exceed the A2 reference receptive field of roughly 6350 samples while remaining small enough for CPU smoke verification.

## Smoke Training Status

| Model | Device | Epochs | Train Batches | Val Batches | Train Loss | Val Loss | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GatedTCN-Small | cpu | 1 | 1 | 1 | 0.198326 | 0.163366 | PASS |
| GatedTCN-Medium | cpu | 1 | 1 | 1 | 0.123419 | 0.858192 | PASS |
| GatedTCN-Large | cpu | 1 | 1 | 1 | 0.139288 | 0.103801 | PASS |

Smoke checkpoints:

- `outputs/tcn_gated/small/checkpoints/best.pt`
- `outputs/tcn_gated/medium/checkpoints/best.pt`
- `outputs/tcn_gated/large/checkpoints/best.pt`

## Formal Training Status

Formal 20-epoch training pending due to CPU-only runtime.

Formal configs are present:

- `configs/tcn_gated/training_formal_small.json`
- `configs/tcn_gated/training_formal_medium.json`
- `configs/tcn_gated/training_formal_large.json`

## Benchmark Status

| Model | Device | RTF | Samples/s | Avg Chunk Latency ms |
| --- | --- | ---: | ---: | ---: |
| GatedTCN-Small | cpu | 0.073829 | 650154.30 | 400.72 |
| GatedTCN-Medium | cpu | 0.178694 | 268615.21 | 970.00 |
| GatedTCN-Large | cpu | 0.336964 | 142448.60 | 1829.17 |

Benchmark JSON files:

- `outputs/tcn_gated/small/benchmark.json`
- `outputs/tcn_gated/medium/benchmark.json`
- `outputs/tcn_gated/large/benchmark.json`

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

`src/ntt/evaluation/compare_models.py` now writes rows for A1 baseline, A2-Lite baseline, A2-Full baseline, GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large.

Current smoke comparison:

| Model | MSE | MAE | ESR | MRSTFT | SNR | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GatedTCN-Small | 0.0206532 | 0.112477 | 1.0597 | 3.90558 | -0.25181 | 0.0738286 |
| GatedTCN-Medium | 0.032953 | 0.150761 | 1.69079 | 4.25608 | -2.28089 | 0.178694 |
| GatedTCN-Large | 0.0195586 | 0.102615 | 1.00353 | 3.50763 | -0.0153178 | 0.336964 |

These are smoke-run metrics, not final formal experiment results.

## Remaining Work

- run formal 20-epoch training on a CUDA-capable runtime
- repeat with multiple seeds
- run Small/Medium/Large ablation on held-out test split
- run activation and loss ablations
- add subjective listening notes after formal training
