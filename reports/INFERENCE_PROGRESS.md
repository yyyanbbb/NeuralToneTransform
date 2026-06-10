# Inference Progress

## Current Status

- A1 inference: completed.
- A2-Lite inference: completed.
- A2-Full inference: completed.
- Inference output verification: PASS.
- Experiment comparison now includes metrics computed from prediction audio.

## Commands

```powershell
.\scripts\inference\run_all_inference.ps1 -ContinueOnError
```

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\inference\verify_inference_outputs.py
```

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav
```

## Outputs

- `outputs/a1_baseline/prediction.wav`
- `outputs/a2_baseline/a2_lite_prediction.wav`
- `outputs/a2_baseline/a2_full_prediction.wav`

## Results

| Output | File Size | Sample Rate | Duration | Peak Amplitude |
| --- | --- | --- | --- | --- |
| `outputs/a1_baseline/prediction.wav` | 18240016 bytes | 48000 | 189.999708 s | 0.509399 |
| `outputs/a2_baseline/a2_lite_prediction.wav` | 18240016 bytes | 48000 | 189.999708 s | 0.827301 |
| `outputs/a2_baseline/a2_full_prediction.wav` | 18240016 bytes | 48000 | 189.999708 s | 0.647705 |

Verification:

```text
pass_count: 19
fail_count: 0
OVERALL: PASS
```

## Metrics Summary

See `reports/experiment_comparison.md` for the generated A1/A2-Lite/A2-Full table.

## Remaining Work

- Verify inference behavior after longer A2 training.
- Add inference latency / speed benchmark.
- Add plots or waveform comparison.
- Evaluate on held-out data beyond the baseline aligned pair.
