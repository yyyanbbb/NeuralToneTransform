# Final TCN Model Card

## Final Selected Model

GatedTCN-Medium is selected as the final custom TCN model.

## Reason for Selection

- It completed 20-epoch formal GPU training.
- It achieved the best held-out test performance among formally trained custom TCN models.
- It outperformed A1/A2 baselines on Test ESR and Test SNR.
- It has a lower RTF than A1/A2 baselines in the current benchmark.
- Small and Large remain smoke checkpoints and are not selected as final models.

## Architecture

| Field | Value |
|---|---|
| Model | GatedTCN-Medium |
| Channels | 32 |
| Skip Channels | 32 |
| Receptive Field | 4093 |
| Parameters | 167553 |
| Activation | Gated tanh-sigmoid |
| Causality | Causal dilated convolution |
| Residual Connection | Yes |
| Skip Connection | Yes |

## Training

| Field | Value |
|---|---|
| Training Config | configs/tcn_gated/training_formal_medium.json |
| Epochs | 20 |
| Device | cuda |
| CUDA Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Best Epoch | 19 |
| Best Val Loss | 0.00628651725128293 |
| Checkpoint | outputs/tcn_gated/medium/checkpoints/best.pt |

## Held-out Test Metrics

| Metric | Value |
|---|---:|
| Test MSE | 0.0008283822842243348 |
| Test MAE | 0.02012827992587347 |
| Test ESR | 0.06080636323433006 |
| Test MRSTFT | 1.9389014005661012 |
| Test SNR | 12.32761074742885 |

## Inference Speed

| Metric | Value |
|---|---:|
| RTF | 0.01329061198095302 |
| Samples/s | 3611571.842499769 |
| Average Chunk Latency ms | 71.7117828566448 |

## Limitations

- Only GatedTCN-Medium has completed formal 20-epoch training.
- Small and Large are currently smoke checkpoints.
- Final subjective listening evaluation still needs manual human judgment.
- The model is trained on the current aligned dataset and may not generalize to unseen amplifiers or pedals without retraining.
