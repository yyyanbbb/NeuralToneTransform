# Final Experiment Summary

## 1. Goal

This project builds a custom neural audio black-box modeling system for dry-to-wet guitar tone transformation.

The goal is not only to run existing NAM baselines, but also to implement and evaluate a custom TCN-based neural audio model.

## 2. Baselines

The project uses the following baselines:

- A1 baseline
- A2-Lite baseline
- A2-Full baseline

## 3. Custom Models

The project implements the following custom Gated TCN variants:

- GatedTCN-Small
- GatedTCN-Medium
- GatedTCN-Large

## 4. Final Selected Model

GatedTCN-Medium is selected as the final custom model.

Reasons:

- It completed 20-epoch formal GPU training.
- It has a validated formal checkpoint.
- It was evaluated on the held-out test split.
- It achieved strong Test ESR and Test SNR.
- It has a practical inference speed with RTF below 1.
- Small and Large remain smoke checkpoints and are not selected as final models.

## 5. Held-out Test Comparison

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR | RTF | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 | 0.0670541 | Baseline prediction-file mode |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 | 0.0517085 | Baseline prediction-file mode |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 | 0.0708378 | Baseline prediction-file mode |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 | 0.0738286 | Smoke checkpoint, not formal training |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 | 0.0132906 | Formal 20-epoch GPU checkpoint; final selected custom model |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 | 0.336964 | Smoke checkpoint, not formal training |

## 6. Technical Interpretation

Lower ESR means the predicted waveform has lower energy-normalized error against the target wet signal. Higher SNR means the prediction is closer to the target signal. Lower MRSTFT means the model better matches the target in the time-frequency domain. RTF lower than 1 means faster-than-real-time offline inference.

GatedTCN-Medium significantly improves over its earlier CPU smoke checkpoint. In the current experiment, GatedTCN-Medium outperforms A1/A2 baselines on key held-out test metrics such as Test ESR and Test SNR. This result should be interpreted within the current dataset and split.

## 7. Why Medium Instead of Large

GatedTCN-Large has a larger receptive field but currently remains a smoke checkpoint. Large also has higher inference cost in the current benchmark. GatedTCN-Medium has completed formal training and has validated held-out test metrics. Therefore, Medium is the defensible final selected model. Large can be considered future work if more GPU time is available.

## 8. Remaining Work

- Formal train Small and Large if more GPU time is available.
- Run multiple random seeds.
- Test on more amplifier / pedal datasets.
- Add subjective listening tests with human raters.
- Compare against more architectures such as LSTM, WaveNet-like variants, or lightweight Conv-TCN variants.
