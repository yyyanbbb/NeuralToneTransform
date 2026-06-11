# Held-out Test Split Evaluation

## Purpose

Final metrics should be reported on the held-out test split, not only on the full aligned audio. The train split is used for training, the val split is used for checkpoint selection, and the test split is used for final metrics.

GatedTCN-Medium uses the 20-epoch formal GPU checkpoint. GatedTCN-Medium test metrics are based on the 20-epoch formal GPU checkpoint.

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
C:\Users\yanbo\.conda\envs\ntt-gpu-cu128\python.exe .\src\ntt\evaluation\evaluate_test_split.py `
  --metadata data/chunks/metadata.json `
  --tcn-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt `
  --model-name GatedTCN-Medium `
  --out reports/test_metrics_tcn_medium.json `
  --device auto
```

## Results

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR | Checkpoint Status |
|---|---:|---:|---:|---:|---:|---|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 | A1 baseline |
| A2-Lite | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 | A2 Lite baseline |
| A2-Full | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 | A2 Full baseline |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 | Smoke checkpoint, not formal training |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 | Formal 20-epoch GPU checkpoint |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 | Smoke checkpoint, not formal training |

## Notes

- train split is used for training;
- val split is used for checkpoint selection;
- test split is used for final metrics;
- Medium test metrics were recomputed from `outputs/tcn_gated/medium/checkpoints/best.pt` using CUDA;
- Small and Large remain smoke checkpoint results.
