# Held-out Test Split Evaluation

## Purpose

Final metrics should be reported on the held-out test split, not only on the full aligned audio. The train split is used for training, the val split is used for checkpoint selection, and the test split is used for final metrics.

## Commands

A1:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a1_baseline/prediction.wav `
  --model-name A1-baseline `
  --out reports/test_metrics_a1.json
```

A2-Lite:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a2_baseline/a2_lite_prediction.wav `
  --model-name A2-Lite `
  --out reports/test_metrics_a2_lite.json
```

A2-Full:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --target data/aligned/aligned_wet.wav `
  --prediction outputs/a2_baseline/a2_full_prediction.wav `
  --model-name A2-Full `
  --out reports/test_metrics_a2_full.json
```

GatedTCN-Medium:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --tcn-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt `
  --model-name GatedTCN-Medium `
  --out reports/test_metrics_tcn_medium.json `
  --device auto
```

## Results

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR |
|---|---:|---:|---:|---:|---:|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 |
| A2-Lite | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 |
| A2-Full | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 |
| GatedTCN-Medium | 0.0294477 | 0.144289 | 2.63712 | 5.13135 | -3.5008 |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 |

## Notes

- train split is used for training;
- val split is used for checkpoint selection;
- test split is used for final metrics;
- TCN results currently use CPU smoke checkpoints, not formal 20-epoch checkpoints.
