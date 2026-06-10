# Reproducibility Check

## Latest Run

Command:

```powershell
.\.venv-a2\Scripts\python.exe .\scripts\common\verify_reproducibility.py --target all
```

Result:

```text
timestamp: 2026-06-10T19:40:59+08:00
target: all
pass_count: 31
fail_count: 0
OVERALL: PASS
```

## Notes

- A1 PowerShell verification passed with `pass_count: 12`, `fail_count: 0`.
- A2 PowerShell verification passed with `pass_count: 13`, `fail_count: 0`.
- The same `.venv-a2` Python interpreter was used for the shared reproducibility script.
