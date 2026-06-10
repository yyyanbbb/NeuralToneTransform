# Custom TCN Ablation

## Models

| Model | Channels | Layers | Receptive Field | Parameters | Training Status |
|---|---:|---:|---:|---:|---|
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | CPU smoke PASS; formal 20-epoch pending |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | CPU smoke PASS; formal 20-epoch pending due to CPU-only runtime |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | CPU smoke PASS; formal 20-epoch pending |

## Held-out Test Metrics

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR | RTF |
|---|---:|---:|---:|---:|---:|---:|
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 | 0.0738286 |
| GatedTCN-Medium | 0.0294477 | 0.144289 | 2.63712 | 5.13135 | -3.5008 | 0.178694 |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 | 0.336964 |

## Observations

- Small: fastest TCN smoke inference and lowest parameter count.
- Medium: original self-developed baseline variant; formal 20-epoch support is implemented but not run in this CPU-only runtime.
- Large: receptive field is 8189 samples, exceeding the approximate A2 reference of 6350 samples; current smoke checkpoint has the strongest TCN test MSE/SNR among the three but is slower on CPU.

## Remaining Work

- train all variants for 20 epochs
- repeat with multiple seeds
- compare against A1/A2 using formal TCN checkpoints
