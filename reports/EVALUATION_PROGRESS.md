# EVALUATION PROGRESS

## Current Status

Implemented objective metrics:

- MSE
- MAE
- ESR
- Normalized MAE
- SNR dB
- Multi-resolution STFT loss

Implemented comparison table generation:

- Artifact-only A1/A2 comparison.
- Optional prediction-audio metric computation when prediction WAV files are available.
- Missing prediction audio is marked as `TBD`; metrics are not fabricated.

## How to Run

Metrics smoke test:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\metrics.py --pred data/aligned/aligned_dry.wav --target data/aligned/aligned_wet.wav
```

Artifact comparison:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py --artifact-only
```

Future audio-metric comparison:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav
```

## Remaining Work

- Add inference outputs for A1.
- Add separate inference outputs for A2-Lite and A2-Full.
- Compute real audio metrics against held-out target audio.
- Add CPU/runtime measurements for A1, A2-Lite, and A2-Full.

## Latest Smoke Test

Command:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\metrics.py --pred data/aligned/aligned_dry.wav --target data/aligned/aligned_wet.wav
```

Result:

```text
MSE: 0.03225153463
MAE: 0.1291629367
ESR: 1.654796334
Normalized MAE: 1.191940604
SNR dB: -2.187445501
MRSTFT: 4.181783676
```

Artifact comparison command:

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py --artifact-only
```

Result:

```text
comparison report: reports/experiment_comparison.md
```
