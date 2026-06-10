# Experiment Comparison

| Model | Architecture | Channels | Training Version | Output Path | ESR | MRSTFT | File Size | Inference Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 baseline | Legacy NAM WaveNet | 16 / 8 internal channels | neural-amp-modeler 0.12.2 | `outputs/a1_baseline/model.nam` | TBD | TBD | 297454 bytes | Reproduced legacy baseline; CPU training previously completed. |
| A2 baseline | PackedWaveNet / SlimmableContainer | 3 Lite / 8 Full | neural-amp-modeler 0.13.0 | `outputs/a2_baseline/model.nam` | TBD | TBD | 308130 bytes | Smoke training completed; exported SlimmableContainer with submodel channels [3, 8]. |
