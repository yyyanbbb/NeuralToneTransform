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

- It completed 20-epoch formal GPU training before final package selection.
- It has a validated formal checkpoint.
- It was evaluated on the held-out test split.
- It achieved strong Test ESR and Test SNR.
- It has practical inference speed with RTF below 1.
- Small and Large formal ablation results are now available, but they are not automatically selected as final models.
- Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.

## 5. Held-out Test Comparison

| Model | Test MSE | Test MAE | Test ESR | Test MRSTFT | Test SNR | RTF | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 baseline | 0.0253307 | 0.0999642 | 1.7338 | 1.54993 | -1.91394 | 0.0670541 | Baseline prediction-file mode |
| A2-Lite baseline | 0.0161523 | 0.0939756 | 1.08471 | 4.6246 | -0.275191 | 0.0517085 | Baseline prediction-file mode |
| A2-Full baseline | 0.0207692 | 0.0924068 | 1.42595 | 2.27191 | -0.982792 | 0.0708378 | Baseline prediction-file mode |
| GatedTCN-Small | 0.00207262 | 0.0302494 | 0.141307 | 2.11301 | 8.538 | 0.00647809 | Formal 20-epoch GPU checkpoint; fastest TCN variant |
| GatedTCN-Medium | 0.000828382 | 0.0201283 | 0.0608064 | 1.9389 | 12.3276 | 0.0132906 | Formal 20-epoch GPU checkpoint; final selected custom model |
| GatedTCN-Large | 0.000702926 | 0.0178562 | 0.0489031 | 1.86984 | 13.1984 | 0.0500886 | Formal 20-epoch GPU checkpoint; strongest TCN test metrics but higher RTF |

## 6. Technical Interpretation

Lower ESR means the predicted waveform has lower energy-normalized error against the target wet signal. Higher SNR means the prediction is closer to the target signal. Lower MRSTFT means the model better matches the target in the time-frequency domain. RTF lower than 1 means faster-than-real-time offline inference.

All three custom TCN variants now have 20-epoch formal GPU ablation results. Small is fastest but less accurate than Medium and Large. Large has the strongest held-out test metrics, but it is slower than Medium in the current benchmark. Medium remains the final selected custom model because this final package preserves the earlier selected model while documenting the new ablation trade-off. The result should be interpreted within the current dataset and held-out split.

## 7. Why Medium Instead of Large

GatedTCN-Large has a larger receptive field and now has stronger held-out test metrics than Medium, but it also has higher inference cost. GatedTCN-Medium remains a defensible final selected model because it provides a strong accuracy/speed balance and was already selected for the final package. Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency.

## 8. Remaining Work

- Run multiple random seeds.
- Test on more amplifier / pedal datasets.
- Add subjective listening tests with human raters.
- Compare against more architectures such as LSTM, WaveNet-like variants, or lightweight Conv-TCN variants.
