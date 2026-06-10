# Custom TCN vs A1/A2 Comparison

The custom model is integrated into the same comparison path as A1 and A2. The current `GatedTCN-Medium` numbers come from a CPU smoke training run, not a full-quality training run.

| Model | MSE | MAE | ESR | MRSTFT | SNR dB |
| --- | --- | --- | --- | --- | --- |
| A1 baseline | 0.0236522 | 0.0978778 | 1.21357 | 1.11878 | -0.84065 |
| A2-Lite baseline | 0.0188753 | 0.101704 | 0.968476 | 4.18794 | 0.139113 |
| A2-Full baseline | 0.0198791 | 0.0936032 | 1.01998 | 1.94226 | -0.0859018 |
| GatedTCN-Medium | 0.032953 | 0.150761 | 1.69079 | 4.25608 | -2.28089 |

`GatedTCN-Medium` currently has 167553 trainable parameters and a 4093-sample receptive field. The architecture and training/inference/evaluation loop are in place; the next quality step is longer training and model/loss ablation.
