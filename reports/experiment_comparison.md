# Experiment Comparison

This table compares the currently available A1 and A2 baseline artifacts. Audio metrics are reported only when prediction audio is available; otherwise they are marked as TBD.

| Model | Architecture | Channels | Training Version | Output Path | Prediction Path | File Size | MSE | MAE | ESR | MRSTFT | SNR | Inference Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 baseline | legacy NAM WaveNet | single model | neural-amp-modeler==0.12.2 | outputs/a1_baseline/model.nam | outputs/a1_baseline/prediction.wav | 297454 bytes | 0.0236522 | 0.0978778 | 1.21357 | 1.11878 | -0.84065 | legacy NAM baseline artifact; audio metrics computed from prediction file |
| A2-Lite baseline | PackedWaveNet exported as SlimmableContainer | 3 | neural-amp-modeler==0.13.0 | outputs/a2_baseline/model.nam | outputs/a2_baseline/a2_lite_prediction.wav | 308130 bytes | 0.0188753 | 0.101704 | 0.968476 | 4.18794 | 0.139113 | A2 smoke status PASS; SlimmableContainer inspected; audio metrics computed from prediction file |
| A2-Full baseline | PackedWaveNet exported as SlimmableContainer | 8 | neural-amp-modeler==0.13.0 | outputs/a2_baseline/model.nam | outputs/a2_baseline/a2_full_prediction.wav | 308130 bytes | 0.0198791 | 0.0936032 | 1.01998 | 1.94226 | -0.0859018 | A2 smoke status PASS; SlimmableContainer inspected; audio metrics computed from prediction file |
