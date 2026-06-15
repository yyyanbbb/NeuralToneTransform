# Final TCN Model Card

## Final Selected Model

GatedTCN-Medium is selected as the final custom TCN model.

## Reason for Selection

- It completed 20-epoch formal GPU training.
- It achieved strong held-out test performance using the formal checkpoint.
- It outperformed A1/A2 baselines on key test metrics in the current experiment, especially Test ESR and Test SNR.
- It has a lower RTF than A1/A2 baselines in the current benchmark.
- Small and Large now also have formal 20-epoch GPU ablation results, but they are not selected as final models in this package.
- Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.

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
| Prediction | outputs/tcn_gated/medium/prediction.wav |

## Held-out Test Metrics

| Metric | Value |
|---|---:|
| Test MSE | 0.000828382 |
| Test MAE | 0.0201283 |
| Test ESR | 0.0608064 |
| Test MRSTFT | 1.9389 |
| Test SNR | 12.3276 |

## Inference Speed

| Metric | Value |
|---|---:|
| RTF | 0.01329061198095302 |
| Samples/s | 3611571.842499769 |
| Average Chunk Latency ms | 71.7117828566448 |
| Inference Time Seconds | 2.5252123999525793 |

## Comparison Summary

| Model | Test ESR | Test SNR | RTF | Notes |
|---|---:|---:|---:|---|
| A1 baseline | 1.7338 | -1.91394 | 0.0670541 | Baseline prediction-file mode |
| A2-Lite baseline | 1.08471 | -0.275191 | 0.0517085 | Baseline prediction-file mode |
| A2-Full baseline | 1.42595 | -0.982792 | 0.0708378 | Baseline prediction-file mode |
| GatedTCN-Small | 0.141307 | 8.538 | 0.00647809 | Formal 20-epoch GPU checkpoint; fastest TCN variant |
| GatedTCN-Medium | 0.0608064 | 12.3276 | 0.0132906 | Final selected model; formal 20-epoch GPU checkpoint |
| GatedTCN-Large | 0.0489031 | 13.1984 | 0.0500886 | Formal 20-epoch GPU checkpoint; strongest TCN test metrics but higher RTF |

- Lower ESR means lower waveform energy error.
- Higher SNR means the prediction is closer to the target wet signal.
- Lower RTF means faster inference.
- RTF lower than 1 means faster-than-real-time offline inference.

## Limitations

- GatedTCN-Medium remains the final selected custom model for this package.
- Small and Large formal ablation results were added after Medium selection and do not automatically change the final model.
- Final subjective listening evaluation still needs manual human judgment.
- The model is trained on the current aligned dataset and may not generalize to unseen amplifiers, pedals, recording chains, or guitar tones without retraining.
- Current results are based on the current dataset and held-out test split only.
