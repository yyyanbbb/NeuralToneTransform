# Experiment Comparison

This table compares the currently available A1 and A2 baseline artifacts. Audio metrics are reported only when prediction audio is available; otherwise they are marked as TBD.

| Model | Architecture | Channels | Training Version | Output Path | File Size | Smoke Status | MSE | MAE | ESR | MRSTFT | SNR | Inference Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 baseline | legacy NAM WaveNet | single model | neural-amp-modeler==0.12.2 | outputs/a1_baseline/model.nam | 297454 bytes | PASS | TBD | TBD | TBD | TBD | TBD | legacy NAM baseline artifact; prediction audio not evaluated; target audio not available |
| A2 baseline | PackedWaveNet exported as SlimmableContainer | 3 Lite / 8 Full | neural-amp-modeler==0.13.0 | outputs/a2_baseline/model.nam | 308130 bytes | PASS | TBD | TBD | TBD | TBD | TBD | A2 smoke baseline; SlimmableContainer inspected; prediction audio not evaluated; target audio not available |
