# NAM Inference Benchmark Progress

## Purpose

A1/A2 RTF is needed for fair speed comparison with TCN. The NAM benchmark uses the same offline inference path as the A1/A2 prediction scripts.

## Commands

A1:

```powershell
.\.venv\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a1_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a1 `
  --out outputs/a1_baseline/benchmark.json
```

A2-Lite:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a2_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a2-lite `
  --out outputs/a2_baseline/a2_lite_benchmark.json
```

A2-Full:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\inference\benchmark_nam_inference.py `
  --model outputs/a2_baseline/model.nam `
  --input data/aligned/aligned_dry.wav `
  --variant a2-full `
  --out outputs/a2_baseline/a2_full_benchmark.json
```

## Results

| Model | Variant | RTF | Samples/s | Avg Chunk Latency ms |
|---|---|---:|---:|---:|
| A1 baseline | a1 | 0.0670541 | 715840.39 | 91.00 |
| A2-Lite baseline | a2-lite | 0.0517085 | 928280.72 | 70.18 |
| A2-Full baseline | a2-full | 0.0708378 | 677604.35 | 96.14 |

## Notes

All three NAM benchmark commands completed and wrote JSON outputs. These RTF values are now included in `reports/experiment_comparison.md`.
