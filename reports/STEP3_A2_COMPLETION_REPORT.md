# STEP3 A2 Completion Report

## Current Status

Status: A2 smoke baseline completed.

The A2 baseline has been reproduced at smoke-test level. The exported model was inspected successfully as a SlimmableContainer with 3-channel Lite and 8-channel Full submodels.

## Model Inspection Summary

- Model path: `outputs/a2_baseline/model.nam`
- Architecture: `SlimmableContainer`
- Submodels: A2-Lite / 3 channels, A2-Full / 8 channels
- Inspection: PASS
- Inspection report: `reports/a2_model_inspection.md`
- Model size: `308130` bytes

## Packed Config

`scripts/a2/prepare_a2_baseline.py` found and copied the official packed config from the installed package resource:

```text
nam/train/_resources/config_model_packed.json
```

The copied config is stored at:

```text
configs/a2_baseline/model_packed.json
```

The official config contains `PackedWaveNet`, `channels_3`, `channels_8`, and packed export settings.

## Verification Summary

Latest verification is refreshed by:

```powershell
.\scripts\a2\finalize_a2_model.ps1
.\scripts\a2\verify_a2_baseline.ps1
.\.venv-a2\Scripts\python.exe .\scripts\common\verify_reproducibility.py --target all
```

Results are recorded below after each validation run.

Latest PowerShell verification result:

```text
timestamp: 2026-06-10 19:39:51 +08:00
finalize_a2_model.ps1: PASS
verify_a2_baseline.ps1 pass_count: 13
verify_a2_baseline.ps1 fail_count: 0
verify_a2_baseline.ps1 OVERALL: PASS
```

Latest shared Python reproducibility result:

```text
timestamp: 2026-06-10T19:40:59+08:00
pass_count: 31
fail_count: 0
OVERALL: PASS
```

## Historical Issue

Earlier runs encountered a PowerShell `$LASTEXITCODE` handling issue:

```text
The variable '$LASTEXITCODE' cannot be retrieved because it has not been set.
```

This has been moved here as a historical issue and should not be treated as the latest A2 status. The scripts now initialize and read `$global:LASTEXITCODE` when handling external command exit codes.

## Remaining Work

- Run longer A2 training beyond smoke-test epochs.
- Add formal ESR / MRSTFT evaluation.
- Compare A1 vs A2-Lite vs A2-Full.
- Measure CPU usage for Lite and Full inference paths.


## Latest A2 Model Inspection

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
