# Subjective Listening Evaluation Template

## Purpose

This document provides a manual listening evaluation template for comparing target wet audio with A1, A2-Lite, A2-Full, and GatedTCN-Medium.

## Audio Files

| Model | File |
|---|---|
| Target Wet | data/aligned/aligned_wet.wav |
| A1 | outputs/a1_baseline/prediction.wav |
| A2-Lite | outputs/a2_baseline/a2_lite_prediction.wav |
| A2-Full | outputs/a2_baseline/a2_full_prediction.wav |
| GatedTCN-Medium | outputs/tcn_gated/medium/prediction.wav |

## Rating Criteria

Use a 1-5 scale:

| Criterion | Description |
|---|---|
| Tone Similarity | How close the predicted tone is to the target wet tone |
| Dynamic Response | Whether picking strength and transients are preserved |
| Noise / Artifact | Whether unnatural noise, buzz, or digital artifacts appear |
| Frequency Balance | Whether bass/mid/treble balance matches the target |
| Overall Preference | Overall perceived quality |

## Notes

This is a subjective supplement to objective metrics. It should not replace held-out test metrics.
