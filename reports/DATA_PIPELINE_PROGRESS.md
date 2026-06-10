# DATA PIPELINE PROGRESS

## Current Status

- Alignment script implemented.
- Chunking script implemented.
- Dataset loader implemented.
- Verification script implemented.

## How to Run

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\align.py --dry data/raw/baseline_input.wav --wet data/raw/baseline_output.wav --out-dir data/aligned
```

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\chunk_audio.py --dry data/aligned/aligned_dry.wav --wet data/aligned/aligned_wet.wav --out-dir data/chunks --chunk-size 65536 --hop-size 65536 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

```powershell
.\.venv-a2\Scripts\python.exe .\src\ntt\data\dataset.py --metadata data/chunks/metadata.json --split train
```

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\common\verify_data_pipeline.py
```

## Output Files

- `data/aligned/aligned_dry.wav`
- `data/aligned/aligned_wet.wav`
- `data/aligned/alignment_metadata.json`
- `data/chunks/metadata.json`

## Latest Verification

```text
alignment: PASS
estimated_delay_samples: 14
aligned_num_samples: 9119986

chunking: PASS
total_chunks: 139
train_chunks: 111
val_chunks: 13
test_chunks: 15

dataset smoke test: PASS
dataset length: 111
first item dry shape: (1, 65536)
first item wet shape: (1, 65536)
sample rate: 48000

verify_data_pipeline.py:
pass_count: 36
fail_count: 0
OVERALL: PASS
```

## Remaining Work

- Larger real datasets.
- More precise delay estimation.
- Multi-recording merge support.
- Formal training data cleaning.
- Custom model training.
