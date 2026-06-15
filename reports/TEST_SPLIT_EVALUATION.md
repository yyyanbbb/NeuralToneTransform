# Held-out Test Split Evaluation

## Purpose

Final metrics should be reported on the held-out test split, not only on the full aligned audio. The train split is used for training and is not used for final testing. The val split is used for checkpoint selection, and the held-out test split is used for final metrics.

A1/A2 use prediction-file mode. GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large use formal checkpoint mode. GatedTCN-Medium remains the final selected custom TCN model.

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

Gated TCN formal checkpoints:

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
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 | A1 baseline prediction-file mode |
| A2-Lite | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 | A2 Lite prediction-file mode |
| A2-Full | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 | A2 Full prediction-file mode |
| GatedTCN-Small | 0.00207262 | 0.0302494 | 0.141307 | 2.11301 | 8.538 | Formal 20-epoch GPU checkpoint |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 | Formal 20-epoch GPU checkpoint; final selected custom model |
| GatedTCN-Large | 0.000702926 | 0.0178562 | 0.0489031 | 1.86984 | 13.1984 | Formal 20-epoch GPU checkpoint |

## Notes

- train split is used for training;
- val split is used for checkpoint selection;
- held-out test split is used for final metrics;
- A1/A2 metrics use prediction-file mode;
- GatedTCN-Small, GatedTCN-Medium, and GatedTCN-Large metrics use formal checkpoint mode;
- Medium test metrics were computed from `outputs/tcn_gated/medium/checkpoints/best.pt` using CUDA;
- Small and Large formal metrics were added for ablation and do not automatically change final model selection.

## Final Model Test Result

GatedTCN-Medium is the final selected custom TCN model. Its held-out test metrics are computed using the 20-epoch formal GPU checkpoint, not the earlier CPU smoke checkpoint.

Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.
