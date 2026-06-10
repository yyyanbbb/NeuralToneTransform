param(
  [switch]$SkipTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv-a2"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$NamFullExe = Join-Path $VenvDir "Scripts\nam-full.exe"
$LogsDir = Join-Path $ProjectRoot "logs\a2"
$PrepareLog = Join-Path $LogsDir "prepare_a2_baseline.log"
$TrainingLog = Join-Path $LogsDir "a2_training.log"
$ReportPath = Join-Path $ProjectRoot "reports\STEP3_A2_COMPLETION_REPORT.md"
$DataCfg = Join-Path $ProjectRoot "configs\a2_baseline\data.json"
$ModelCfg = Join-Path $ProjectRoot "configs\a2_baseline\model_packed.json"
$LearnCfg = Join-Path $ProjectRoot "configs\a2_baseline\learning.json"
$OutDir = Join-Path $ProjectRoot "outputs\a2_baseline"
$ModelOut = Join-Path $OutDir "model.nam"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportPath) | Out-Null
"" | Set-Content -LiteralPath $PrepareLog -Encoding UTF8
"" | Set-Content -LiteralPath $TrainingLog -Encoding UTF8

function Write-LogLine {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message
  )
  $Message | Tee-Object -FilePath $LogPath -Append
}

function Add-A2ReportFailure {
  param(
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(Mandatory = $true)][string]$ErrorSummary
  )
  $Content = @"
# STEP3 A2 COMPLETION REPORT

Status: A2 not completed.

## Latest Failure

- timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')
- failed_command: ``$Command``
- error_summary: $ErrorSummary
- training_log: logs/a2/a2_training.log

## Suggested Next Steps

1. Confirm neural-amp-modeler==0.13.0 supports the current Python version.
2. Confirm configs/a2_baseline/model_packed.json matches the official PackedWaveNet schema.
3. Re-run .\scripts\a2\run_a2_baseline.ps1 after resolving the dependency or schema error.
"@
  $Content | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}

function Invoke-LoggedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][scriptblock]$Command,
    [Parameter(Mandatory = $true)][string]$FailureMessage,
    [Parameter(Mandatory = $true)][string]$CommandText
  )
  $global:LASTEXITCODE = 0
  & $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
  $ExitCode = $global:LASTEXITCODE
  if ($ExitCode -ne 0) {
    $Message = "$FailureMessage (exit code $ExitCode)"
    Write-LogLine -LogPath $LogPath -Message "ERROR: $Message"
    Add-A2ReportFailure -Command $CommandText -ErrorSummary $Message
    exit 1
  }
}

function Invoke-CmdLoggedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$CommandText,
    [Parameter(Mandatory = $true)][string]$FailureMessage
  )
  $global:LASTEXITCODE = 0
  & cmd.exe /d /c $CommandText | Tee-Object -FilePath $LogPath -Append
  $ExitCode = $global:LASTEXITCODE
  if ($ExitCode -ne 0) {
    $Message = "$FailureMessage (exit code $ExitCode)"
    Write-LogLine -LogPath $LogPath -Message "ERROR: $Message"
    Add-A2ReportFailure -Command $CommandText -ErrorSummary $Message
    exit 1
  }
}

try {
  Write-LogLine -LogPath $TrainingLog -Message "=== A2 PackedWaveNet Training ==="
  Write-LogLine -LogPath $TrainingLog -Message "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine -LogPath $TrainingLog -Message "project root: $ProjectRoot"

  if (-not (Test-Path $Python)) {
    Write-LogLine -LogPath $TrainingLog -Message "Creating .venv-a2..."
    $CreateCommand = "py -3 -m venv .venv-a2"
    $global:LASTEXITCODE = 0
    & py -3 -m venv ".venv-a2" 2>&1 | Tee-Object -FilePath $TrainingLog -Append
    if ($global:LASTEXITCODE -ne 0) {
      $CreateCommand = "python -m venv .venv-a2"
      Invoke-LoggedCommand -LogPath $TrainingLog -Command { & python -m venv ".venv-a2" } -FailureMessage "Failed to create .venv-a2" -CommandText $CreateCommand
    }
  }

  Invoke-LoggedCommand -LogPath $TrainingLog -Command { & $Python -m pip install --upgrade pip } -FailureMessage "Failed to upgrade pip in .venv-a2" -CommandText "$Python -m pip install --upgrade pip"
  Invoke-LoggedCommand -LogPath $TrainingLog -Command { & $Pip install -r ".\requirements-a2.txt" } -FailureMessage "Failed to install requirements-a2.txt" -CommandText "$Pip install -r requirements-a2.txt"
  Invoke-LoggedCommand -LogPath $TrainingLog -Command {
    & $Python ".\scripts\common\print_dependency_versions.py" torch torchaudio librosa matplotlib tensorboard soundfile scipy numpy pandas neural-amp-modeler
  } -FailureMessage "Failed to capture A2 dependency versions." -CommandText "$Python scripts/common/print_dependency_versions.py ..."

  Invoke-LoggedCommand -LogPath $PrepareLog -Command { & $Python ".\scripts\a2\prepare_a2_baseline.py" } -FailureMessage "A2 baseline preparation failed" -CommandText "$Python scripts/a2/prepare_a2_baseline.py"

  if ($SkipTrain) {
    Write-LogLine -LogPath $TrainingLog -Message "SkipTrain enabled. Prepared A2 configs without running packed training."
    if (Test-Path $ModelOut) {
      $global:LASTEXITCODE = 0
      & ".\scripts\a2\finalize_a2_model.ps1" 2>&1 | Tee-Object -FilePath $TrainingLog -Append
      $FinalizeExitCode = $global:LASTEXITCODE
      if ($FinalizeExitCode -ne 0) {
        $Message = "A2 finalize step failed with exit code $FinalizeExitCode"
        Write-LogLine -LogPath $TrainingLog -Message "ERROR: $Message"
        Add-A2ReportFailure -Command ".\scripts\a2\finalize_a2_model.ps1" -ErrorSummary $Message
        exit 1
      }
    }
    exit 0
  }

  if (Test-Path $NamFullExe) {
    $TrainingCommand = ('chcp 65001>nul && "{0}" "{1}" "{2}" "{3}" "{4}" 2>&1' -f $NamFullExe, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
  } else {
    $TrainingCommand = ('chcp 65001>nul && "{0}" -m nam.train.full "{1}" "{2}" "{3}" "{4}" 2>&1' -f $Python, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
  }
  Write-LogLine -LogPath $TrainingLog -Message "training command: $TrainingCommand"
  Invoke-CmdLoggedCommand -LogPath $TrainingLog -CommandText $TrainingCommand -FailureMessage "A2 packed training failed"

  $global:LASTEXITCODE = 0
  & ".\scripts\a2\finalize_a2_model.ps1" 2>&1 | Tee-Object -FilePath $TrainingLog -Append
  $FinalizeExitCode = $global:LASTEXITCODE
  if ($FinalizeExitCode -ne 0) {
    Add-A2ReportFailure -Command ".\scripts\a2\finalize_a2_model.ps1" -ErrorSummary "A2 finalize step failed with exit code $FinalizeExitCode"
    exit 1
  }
}
catch {
  Write-LogLine -LogPath $TrainingLog -Message "ERROR: $($_.Exception.Message)"
  Add-A2ReportFailure -Command ".\scripts\a2\run_a2_baseline.ps1" -ErrorSummary $_.Exception.Message
  exit 1
}
