# Custom TCN vs A1/A2 Comparison

The custom models are integrated into the same comparison path as A1 and A2. The current GatedTCN rows come from 20-epoch formal GPU checkpoints.

| Model | MSE | MAE | ESR | MRSTFT | SNR dB |
| --- | --- | --- | --- | --- | --- |
| A1 baseline | 0.0236522 | 0.0978778 | 1.21357 | 1.11878 | -0.84065 |
| A2-Lite baseline | 0.0188753 | 0.101704 | 0.968476 | 4.18794 | 0.139113 |
| A2-Full baseline | 0.0198791 | 0.0936032 | 1.01998 | 1.94226 | -0.0859018 |
| GatedTCN-Small | 0.00326873 | 0.0379112 | 0.167716 | 1.73089 | 7.75426 |
| GatedTCN-Medium | 0.00123466 | 0.0234437 | 0.0633493 | 1.53635 | 11.9826 |
| GatedTCN-Large | 0.00109394 | 0.0218652 | 0.0561293 | 1.51056 | 12.5081 |

`GatedTCN-Medium` currently has 167553 trainable parameters and a 4093-sample receptive field. It is the final selected custom TCN model for the current project state. Large formal result may be considered in future model selection, but this package keeps GatedTCN-Medium as the final selected model for consistency. Final claims should use the held-out test metrics in `reports/FINAL_EXPERIMENT_SUMMARY.md`; these full-file metrics are diagnostic.
