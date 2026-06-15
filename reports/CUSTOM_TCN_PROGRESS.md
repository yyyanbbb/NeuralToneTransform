# Custom Gated TCN Progress

## Current Status

GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large have completed 20-epoch formal GPU training. Prediction audio, held-out test metrics, inference benchmarks, figures, and the comparison table have been regenerated from the current checkpoints.

GatedTCN-Medium remains the final selected custom model for this package. Small and Large formal results strengthen the ablation study but do not automatically change final model selection.

## Formal Training

| Model | Device | Epochs | Best Epoch | Best Val Loss | Training Config | Status |
|---|---|---:|---:|---:|---|---|
| GatedTCN-Small | cuda | 20 | 16 | 0.0130615 | `configs/tcn_gated/training_formal_small.json` | Formal 20-epoch GPU checkpoint |
| GatedTCN-Medium | cuda | 20 | 19 | 0.00628652 | `configs/tcn_gated/training_formal_medium.json` | Formal 20-epoch GPU checkpoint; final selected custom model |
| GatedTCN-Large | cuda | 20 | 18 | 0.00426024 | `configs/tcn_gated/training_formal_large.json` | Formal 20-epoch GPU checkpoint |

## Model Configurations

| Model | Channels | Layers | Receptive Field | Parameters | Config |
|---|---:|---:|---:|---:|---|
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | `configs/tcn_gated/small.json` |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | `configs/tcn_gated/medium.json` |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | `configs/tcn_gated/large.json` |

Large exceeds the A2 reference receptive field of roughly 6350 samples.

## Held-out Test Evaluation

Held-out test split evaluation support is implemented in `src/ntt/evaluation/evaluate_test_split.py`.

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR |
|---|---:|---:|---:|---:|---:|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 |
| GatedTCN-Small | 0.00207262 | 0.0302494 | 0.141307 | 2.11301 | 8.538 |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 |
| GatedTCN-Large | 0.000702926 | 0.0178562 | 0.0489031 | 1.86984 | 13.1984 |

## Benchmark Status

TCN benchmark support is implemented in `src/ntt/tcn/benchmark.py`. A1/A2 NAM benchmark support is implemented in `scripts/inference/benchmark_nam_inference.py`.

| Model | Device / Env | RTF | Samples/s | Avg Chunk Latency ms |
|---|---|---:|---:|---:|
| A1 baseline | `.venv` | 0.0670541 | 715840.39 | 91.00 |
| A2-Lite baseline | `.venv-a2` | 0.0517085 | 928280.72 | 70.18 |
| A2-Full baseline | `.venv-a2` | 0.0708378 | 677604.35 | 96.14 |
| GatedTCN-Small | cuda | 0.00647809 | 7409595.68 | 34.82 |
| GatedTCN-Medium | cuda | 0.0132906 | 3611571.84 | 71.71 |
| GatedTCN-Large | cuda | 0.0500886 | 958302.56 | 271.47 |

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

`reports/experiment_comparison.md` has been regenerated from current A1/A2 predictions, formal TCN predictions, held-out test metric JSON files, and benchmark JSON files.

## Model Selection Note

Large formal result is competitive, but GatedTCN-Medium remains the selected final model unless manually changed. Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.

## Remaining Work

- repeat with multiple seeds
- add subjective listening notes
- evaluate on more amplifier or pedal datasets
