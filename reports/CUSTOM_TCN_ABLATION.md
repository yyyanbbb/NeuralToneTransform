# Custom TCN Ablation

## Models

| Model | Channels | Layers | Receptive Field | Parameters | Training Status | Test MSE | Test ESR | Test MRSTFT | Test SNR | RTF |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | Smoke checkpoint, not formal training | 0.0154348 | 1.08549 | 4.53482 | -0.311414 | 0.0738286 |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | Formal 20-epoch GPU checkpoint | 0.000828382 | 0.0608064 | 1.9389 | 12.3276 | 0.0132906 |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | Smoke checkpoint, not formal training | 0.0133959 | 0.903768 | 4.07872 | 0.448861 | 0.336964 |

## Formal Medium Result

| Field | Value |
|---|---|
| Device | cuda |
| CUDA Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Epochs | 20 |
| Best Epoch | 19 |
| Best Val Loss | 0.00628652 |
| Training Time Seconds | 125.642443 |
| Checkpoint | `outputs/tcn_gated/medium/checkpoints/best.pt` |
| Prediction | `outputs/tcn_gated/medium/prediction.wav` |
| Test Metrics | `reports/test_metrics_tcn_medium.json` |
| Benchmark | `outputs/tcn_gated/medium/benchmark.json` |

## Observations

- Small: fastest previous CPU smoke TCN inference and lowest parameter count, but still not a formal checkpoint.
- Medium: completed 20-epoch formal GPU training and now has the strongest held-out test MSE, ESR, and SNR among the current TCN variants.
- Large: receptive field is 8189 samples, exceeding the approximate A2 reference of 6350 samples, but the current Large result is still a CPU smoke checkpoint.

## Remaining Work

- train Small and Large for 20 epochs if formal ablation coverage is needed
- repeat Medium with multiple seeds
- compare subjective listening notes against A1/A2 and the formal Medium checkpoint
