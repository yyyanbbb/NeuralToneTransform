param(
  [switch]$SkipTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$LogsDir = Join-Path $ProjectRoot "logs\a1"
$PrepareLog = Join-Path $LogsDir "prepare_a1_baseline.log"
$TrainingLog = Join-Path $LogsDir "a1_training.log"
$DataCfg = Join-Path $ProjectRoot "configs\a1_baseline\data.json"
$ModelCfg = Join-Path $ProjectRoot "configs\a1_baseline\model.json"
$LearnCfg = Join-Path $ProjectRoot "configs\a1_baseline\learning.json"
$OutDir = Join-Path $ProjectRoot "outputs\a1_baseline"
$ModelOut = Join-Path $OutDir "model.nam"
$NamFullExe = Join-Path $ProjectRoot ".venv\Scripts\nam-full.exe"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $PrepareLog -Encoding UTF8
"" | Set-Content -LiteralPath $TrainingLog -Encoding UTF8

function Write-LogLine {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message
  )
  $Message | Tee-Object -FilePath $LogPath -Append
}

function Invoke-LoggedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][scriptblock]$Command,
    [Parameter(Mandatory = $true)][string]$FailureMessage
  )
  $global:LASTEXITCODE = 0
  & $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
  $ExitCode = $global:LASTEXITCODE
  if ($ExitCode -ne 0) {
    Write-LogLine -LogPath $LogPath -Message "ERROR: $FailureMessage (exit code $ExitCode)"
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
    Write-LogLine -LogPath $LogPath -Message "ERROR: $FailureMessage (exit code $ExitCode)"
    exit 1
  }
}

try {
  if (-not (Test-Path $Python)) {
    throw ".venv not found. Please run scripts/setup_step1.ps1 first."
  }

  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"

  Write-LogLine -LogPath $TrainingLog -Message "=== A1 NAM Training ==="
  Write-LogLine -LogPath $TrainingLog -Message "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine -LogPath $TrainingLog -Message "python executable: $Python"
  Write-LogLine -LogPath $TrainingLog -Message "project root: $ProjectRoot"
  Write-LogLine -LogPath $TrainingLog -Message "data config path: $DataCfg"
  Write-LogLine -LogPath $TrainingLog -Message "model config path: $ModelCfg"
  Write-LogLine -LogPath $TrainingLog -Message "learning config path: $LearnCfg"
  Write-LogLine -LogPath $TrainingLog -Message "output directory: $OutDir"

  Invoke-LoggedCommand -LogPath $TrainingLog -Command {
    & $Python ".\scripts\common\print_dependency_versions.py" torch torchaudio librosa matplotlib tensorboard soundfile scipy numpy pandas neural-amp-modeler
  } -FailureMessage "Failed to capture A1 dependency versions."

  Write-LogLine -LogPath $PrepareLog -Message "=== A1 Prepare NAM Baseline ==="
  Invoke-LoggedCommand -LogPath $PrepareLog -Command { & $Python ".\scripts\a1\prepare_a1_baseline.py" } -FailureMessage "A1 baseline preparation failed"

  if (-not $SkipTrain) {
    Write-LogLine -LogPath $TrainingLog -Message "Starting A1 NAM baseline training..."
    if (Test-Path $NamFullExe) {
      $TrainingCommand = ('chcp 65001>nul && "{0}" "{1}" "{2}" "{3}" "{4}" 2>&1' -f $NamFullExe, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
      Write-LogLine -LogPath $TrainingLog -Message "training command: $TrainingCommand"
      Invoke-CmdLoggedCommand -LogPath $TrainingLog -CommandText $TrainingCommand -FailureMessage "A1 training failed"
    } else {
      $TrainingCommand = ('chcp 65001>nul && "{0}" -m nam.train.full "{1}" "{2}" "{3}" "{4}" 2>&1' -f $Python, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
      Write-LogLine -LogPath $TrainingLog -Message "training command: $TrainingCommand"
      Invoke-CmdLoggedCommand -LogPath $TrainingLog -CommandText $TrainingCommand -FailureMessage "A1 training failed"
    }
  } else {
    Write-LogLine -LogPath $TrainingLog -Message "SkipTrain enabled. Reusing existing outputs for validation only."
  }

  $global:LASTEXITCODE = 0
  & ".\scripts\a1\finalize_a1_model.ps1" 2>&1 | Tee-Object -FilePath $TrainingLog -Append
  $FinalizeExitCode = $global:LASTEXITCODE
  if ($FinalizeExitCode -ne 0) {
    throw "A1 finalize step failed with exit code $FinalizeExitCode"
  }

  Write-LogLine -LogPath $TrainingLog -Message "A1 baseline flow finished. Canonical model: $ModelOut"
}
catch {
  Write-LogLine -LogPath $PrepareLog -Message "ERROR: $($_.Exception.Message)"
  Write-LogLine -LogPath $TrainingLog -Message "ERROR: $($_.Exception.Message)"
  exit 1
}
