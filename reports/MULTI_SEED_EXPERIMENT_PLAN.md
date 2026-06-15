# Multi-seed Experiment Plan

## Purpose

Multiple random seeds can test whether the final result is stable or only caused by a lucky initialization.

## Planned Seeds

| Seed | Model | Status |
|---:|---|---|
| 42 | GatedTCN-Medium | Completed |
| 43 | GatedTCN-Medium | Planned |
| 44 | GatedTCN-Medium | Planned |

## Metrics to Compare

- Test MSE
- Test ESR
- Test MRSTFT
- Test SNR
- RTF
- Best validation loss

## Current Decision

Multi-seed training is not required for the current final package. It is listed as future work unless additional GPU time is available.
