# A2 Model Inspection

- timestamp: 2026-06-10T19:39:39+08:00
- model_path: outputs/a2_baseline/model.nam
- file_size_bytes: 308130
- first_32_bytes_hex: 7b2276657273696f6e223a2022302e372e30222c20226d65746164617461223a
- parsed_as_json: True
- architecture: SlimmableContainer
- submodel_channels: [3, 8]

## Keyword Checks

- SlimmableContainer: FOUND
- WaveNet: FOUND
- channels: FOUND
- 3: FOUND
- 8: FOUND

## Readable String Snippets

- `{"version": "0.7.0", "metadata": {"date": {"year": 2026, "month": 6, "day": 10, "hour": 1, "minute": 32, "second": 55}, "loudness": -18.2569580078125, "gain": 0`

## Parsed JSON Top Level

- `version`
- `metadata`
- `architecture`
- `config`
- `weights`
- `sample_rate`

SlimmableContainer observed: YES
OVERALL: PASS
