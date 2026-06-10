# Custom TCN Ablation

## Models

| Model | Channels | Layers | Receptive Field | Parameters | Training Status |
|---|---:|---:|---:|---:|---|
| GatedTCN-Small | 16 | 12 | 2053 | 25665 | CPU smoke PASS; formal 20-epoch pending |
| GatedTCN-Medium | 32 | 20 | 4093 | 167553 | CPU smoke PASS; formal 20-epoch pending |
| GatedTCN-Large | 48 | 22 | 8189 | 412225 | CPU smoke PASS; formal 20-epoch pending |

## Metrics

| Model | MSE | MAE | ESR | MRSTFT | SNR | RTF |
|---|---:|---:|---:|---:|---:|---:|
| GatedTCN-Small | 0.0206532 | 0.112477 | 1.0597 | 3.90558 | -0.25181 | 0.0738286 |
| GatedTCN-Medium | 0.032953 | 0.150761 | 1.69079 | 4.25608 | -2.28089 | 0.178694 |
| GatedTCN-Large | 0.0195586 | 0.102615 | 1.00353 | 3.50763 | -0.0153178 | 0.336964 |

## Observations

- Small: fastest smoke inference and lowest parameter count; current smoke metrics are competitive with Large on MSE but are not formal results.
- Medium: original self-developed baseline variant; CPU smoke run verifies the full training/inference/evaluation loop.
- Large: receptive field is 8189 samples, exceeding the approximate A2 reference of 6350 samples, with slower CPU RTF as expected.

## Remaining Work

- train all variants for 20 epochs
- repeat with multiple seeds
- compare against A1/A2 on held-out test split
