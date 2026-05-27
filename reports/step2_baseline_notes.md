# Step2 Baseline Notes (NAM 0.12.2)

Date: 2026-04-23

## Official training entry points

- GUI trainer: `nam`
- Full-featured CLI trainer: `nam-full`
- This project uses full-featured CLI equivalent:
  - `python -m nam.train.full.main <data.json> <model.json> <learning.json> <output_dir>`

## Input requirements (from official docs)

- Paired `input`/`output` WAV files
- Same sample rate / bit depth / length
- If using standardized files, common setting is 48kHz + 24-bit
- Delay must be provided in `data.json` when needed (`common.delay`)

## Config structure

Training uses 3 config files:

1. Data config (`nam_full_configs/data/...`)
2. Model config (`nam_full_configs/models/...`)
3. Learning config (`nam_full_configs/learning/...`)

## Expected outputs

- Output directory with checkpoints and logs
- Final exported model file: `model.nam`

## Inference side

- NAM ecosystem separates:
  - Training/export (`neural-amp-modeler`)
  - Real-time playback (separate plugin repo)
