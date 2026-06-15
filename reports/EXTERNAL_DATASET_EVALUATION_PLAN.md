# External Dataset Evaluation Plan

## Purpose

The current model is trained and evaluated on the current aligned dry/wet dataset. External datasets are needed to test generalization.

## Candidate Extensions

- Additional amplifier captures
- Additional overdrive / distortion pedals
- Different guitars or pickup types
- Different gain levels
- Different recording interfaces

## Required Pipeline

1. Collect new dry/wet pair
2. Run alignment
3. Chunk into train/val/test
4. Run inference or retrain
5. Evaluate held-out test metrics
6. Add subjective listening evaluation

## Current Decision

External dataset evaluation is not included in the current final package. It remains future work.
