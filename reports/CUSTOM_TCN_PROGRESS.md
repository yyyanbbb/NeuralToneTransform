# Custom Gated TCN Progress

## Current Status

The Custom Gated TCN implementation is present under `src/ntt/tcn/`. A CPU smoke training run completed with the medium model, produced `best.pt` and `last.pt`, and full-file chunked inference generated `outputs/tcn_gated/medium/prediction.wav`.

The run used `configs/tcn_gated/training_smoke.json` because the full `configs/tcn_gated/training.json` medium epoch was too slow in this CPU-only environment. The requested formal training config remains unchanged for longer runs.

## Architecture

- causal dilated convolution with explicit left-only padding
- gated tanh-sigmoid activation in `GatedTCNBlock`
- residual connection inside each block
- skip connection aggregation across all blocks
- receptive field: 4093 samples for `GatedTCN-Medium`
- parameter count: 167553 trainable parameters

## Training

- device: cpu
- epochs: 1 smoke epoch
- batch size: 1
- train batches: 1
- val batches: 1
- loss weights: MSE 1.0, ESR 0.1, MRSTFT 0.001
- train total loss: 0.12341918796300888
- train MSE: 0.024810779839754105
- train ESR: 0.9561054706573486
- train MRSTFT: 2.9978506565093994
- val total loss: 0.8581922650337219
- val MSE: 0.01866505667567253
- val ESR: 8.331891059875488
- val MRSTFT: 6.338040828704834
- best checkpoint: `outputs/tcn_gated/medium/checkpoints/best.pt`

## Inference

- input path: `data/aligned/aligned_dry.wav`
- output path: `outputs/tcn_gated/medium/prediction.wav`
- prediction length: 9119986 samples
- sample rate: 48000 Hz
- device: cpu
- chunk size: 262144 samples

## Comparison with A1/A2

`src/ntt/evaluation/compare_models.py` now accepts Custom TCN arguments and writes a `GatedTCN-Medium` row to `reports/experiment_comparison.md` with MSE, MAE, ESR, MRSTFT, SNR, parameter count, and receptive field when the prediction and checkpoint are available.

Current smoke comparison:

- MSE: 0.032953
- MAE: 0.150761
- ESR: 1.69079
- MRSTFT: 4.25608
- SNR: -2.28089 dB

## Remaining Work

- train for more epochs
- tune model size
- run Small/Medium/Large ablation
- run activation/loss ablation
- add inference speed benchmark
