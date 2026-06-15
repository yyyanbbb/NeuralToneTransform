# Subjective Listening Evaluation Template

## Purpose

This document provides a manual listening evaluation template for comparing target wet audio with A1, A2-Lite, A2-Full, and the final selected GatedTCN-Medium model.

This subjective evaluation is a supplement to objective metrics such as MSE, ESR, MRSTFT, SNR, and RTF. It should not replace held-out test evaluation.

## Current Status

Subjective listening evaluation has not yet been manually completed. The table below is prepared for human evaluation and should be filled after listening to the target wet audio and each model prediction under the same playback conditions.

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
| A1 Baseline | TBD | TBD | TBD | TBD | TBD | Pending manual listening |
| A2-Lite Baseline | TBD | TBD | TBD | TBD | TBD | Pending manual listening |
| A2-Full Baseline | TBD | TBD | TBD | TBD | TBD | Pending manual listening |
| GatedTCN-Medium | TBD | TBD | TBD | TBD | TBD | Pending manual listening |

## Listening Notes

- Target wet tone characteristics:
- A1 notes:
- A2-Lite notes:
- A2-Full notes:
- GatedTCN-Medium notes:
- Final subjective conclusion:

## Manual Evaluation Procedure

1. Use the same playback device and volume for all audio files.
2. Listen to the target wet audio first.
3. Listen to each prediction without changing volume.
4. Rate each model using the 1-5 scale.
5. If possible, conduct blind listening by hiding model names during rating.
6. Use objective metrics only as supporting evidence, not as a substitute for listening.

## Notes

This subjective evaluation is a supplement to objective metrics such as MSE, ESR, MRSTFT, SNR, and RTF. It should not replace held-out test evaluation.
