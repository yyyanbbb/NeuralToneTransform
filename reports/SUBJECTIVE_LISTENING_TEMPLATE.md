# Subjective Listening Evaluation Template

## Purpose

This document provides a manual listening evaluation template for comparing target wet audio with A1, A2-Lite, A2-Full, and the final selected GatedTCN-Medium model.

This subjective evaluation is a supplement to objective metrics such as MSE, ESR, MRSTFT, SNR, and RTF. It should not replace held-out test evaluation.

## Audio Files

| Model | File |
|---|---|
| Target Wet | data/aligned/aligned_wet.wav |
| A1 Baseline | outputs/a1_baseline/prediction.wav |
| A2-Lite Baseline | outputs/a2_baseline/a2_lite_prediction.wav |
| A2-Full Baseline | outputs/a2_baseline/a2_full_prediction.wav |
| GatedTCN-Medium | outputs/tcn_gated/medium/prediction.wav |

## Rating Scale

Use a 1-5 scale:

| Score | Meaning |
|---:|---|
| 1 | Very poor |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent |

## Rating Criteria

| Criterion | Description |
|---|---|
| Tone Similarity | How close the predicted tone is to the target wet tone |
| Dynamic Response | Whether picking strength and transients are preserved |
| Noise / Artifact | Whether unnatural noise, buzz, clicks, or digital artifacts appear |
| Frequency Balance | Whether bass, mid, and treble balance matches the target |
| Overall Preference | Overall perceived quality |

## Evaluation Table

| Model | Tone Similarity | Dynamic Response | Noise / Artifact | Frequency Balance | Overall Preference | Notes |
|---|---:|---:|---:|---:|---:|---|
| A1 Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| A2-Lite Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| A2-Full Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| GatedTCN-Medium | TBD | TBD | TBD | TBD | TBD | TBD |

## Listening Notes

- Target wet tone characteristics:
- A1 notes:
- A2-Lite notes:
- A2-Full notes:
- GatedTCN-Medium notes:
- Final subjective conclusion:

## Notes

This listening test should be performed using the same playback device and volume level where possible. The listener should avoid looking at the model name during rating if blind evaluation is possible.
