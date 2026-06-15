# Final Experiment Summary

## 1. Goal

This project builds a custom neural audio black-box modeling system for dry-to-wet guitar tone transformation.

## 2. Baselines

- A1 baseline
- A2-Lite baseline
- A2-Full baseline

## 3. Custom Model

- GatedTCN-Small
- GatedTCN-Medium
- GatedTCN-Large

## 4. Final Selected Model

GatedTCN-Medium is selected as the final custom model.

## 5. Held-out Test Comparison

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR | RTF | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 | 0.0670541 | Baseline prediction-file mode |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 | 0.0517085 | Baseline prediction-file mode |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 | 0.0708378 | Baseline prediction-file mode |
| GatedTCN-Small | 0.0154348 | 0.0996688 | 1.08549 | 4.53482 | -0.311414 | 0.0738286 | Smoke checkpoint, not formal training |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 | 0.0132906 | Formal 20-epoch GPU checkpoint; final selected custom model |
| GatedTCN-Large | 0.0133959 | 0.0873261 | 0.903768 | 4.07872 | 0.448861 | 0.336964 | Smoke checkpoint, not formal training |

## 6. Interpretation

Lower ESR means the waveform energy error is lower. Higher SNR means the prediction is closer to the target. RTF lower than 1 means faster-than-real-time offline inference. On the current aligned dataset and held-out test split, the Medium formal checkpoint significantly improves over its previous smoke checkpoint and outperforms A1/A2 on key test metrics in this experiment.

## 7. Why Medium Instead of Large

Large currently remains a smoke checkpoint. It has a larger receptive field, but it also has much higher inference cost in the current benchmark. Medium has completed formal 20-epoch GPU training and gives better validated held-out results. Therefore, Medium is the defensible final model for the current project state.

## 8. Remaining Work

- Formal train Small and Large if more GPU time is available.
- Run multiple random seeds.
- Use more amplifier / pedal datasets.
- Add listening test with human raters.
