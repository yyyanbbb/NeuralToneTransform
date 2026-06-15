# Custom TCN Ablation

## Models

| Model | Channels | Layers | Receptive Field | Parameters | Training Status | Test MSE | Test ESR | Test MRSTFT | Test SNR | RTF |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | Formal 20-epoch GPU checkpoint | 0.00207262 | 0.141307 | 2.11301 | 8.538 | 0.00647809 |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | Formal 20-epoch GPU checkpoint; final selected custom model | 0.000828382 | 0.0608064 | 1.9389 | 12.3276 | 0.0132906 |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | Formal 20-epoch GPU checkpoint | 0.000702926 | 0.0489031 | 1.86984 | 13.1984 | 0.0500886 |

## Formal Training Results

| Field | GatedTCN-Small | GatedTCN-Medium | GatedTCN-Large |
|---|---:|---:|---:|
| Device | cuda | cuda | cuda |
| Epochs | 20 | 20 | 20 |
| Best Epoch | 16 | 19 | 18 |
| Best Val Loss | 0.0130615 | 0.00628652 | 0.00426024 |
| Parameters | 25665 | 167553 | 412225 |
| Receptive Field | 2053 | 4093 | 8189 |
| RTF | 0.00647809 | 0.0132906 | 0.0500886 |

## Observations

- Small is the fastest formal TCN variant with RTF 0.00647809, but it has weaker held-out accuracy than Medium and Large: Test ESR 0.141307 and Test SNR 8.538.
- Medium remains the selected final custom model for this package. It gives a strong accuracy/speed balance with Test ESR 0.0608064, Test SNR 12.3276, and RTF 0.0132906.
- Large achieved the strongest held-out test metrics in this ablation, with Test ESR 0.0489031 and Test SNR 13.1984, but its RTF 0.0500886 is higher than Small and Medium.
- Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.
- Small, Medium, and Large now have 20-epoch formal GPU results, so this ablation is no longer limited to smoke checkpoint comparison.

## Remaining Work

- repeat Medium and other variants with multiple seeds
- compare subjective listening notes against A1/A2 and all formal TCN checkpoints
- evaluate on additional amplifier or pedal datasets before making generalization claims
