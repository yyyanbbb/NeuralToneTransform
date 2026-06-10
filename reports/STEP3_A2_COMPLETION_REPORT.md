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

## Latest Verification Summary

```text
finalize_a2_model.ps1: PASS
verify_a2_baseline.ps1: PASS
verify_a2_baseline.ps1 pass_count: 13
verify_a2_baseline.ps1 fail_count: 0
```

Shared reproducibility:

```text
pass_count: 31
fail_count: 0
OVERALL: PASS
```

## Remaining Work

- Run longer A2 training beyond smoke-test epochs.
- Add final held-out inference metrics after longer training.
- Compare A1 vs A2-Lite vs A2-Full on CPU/runtime.
- Add plots or waveform comparison.

## Failure Log

### Previous historical issue

- Failed command: `.\scripts\a2\run_a2_baseline.ps1`
- Error summary: Earlier runs encountered a PowerShell `$LASTEXITCODE` handling issue.
- Exit code: `1`
- Notes: This failure log does not invalidate the previously completed A2 smoke baseline unless the canonical model artifact is removed or model inspection no longer passes.
