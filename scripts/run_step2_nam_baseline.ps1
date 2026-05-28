param(
  [switch]$SkipTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$LogsDir = Join-Path $ProjectRoot "logs"
$PrepareLog = Join-Path $LogsDir "step2_prepare_nam_baseline.log"
$TrainingLog = Join-Path $LogsDir "step2_nam_training.log"
$DataCfg = Join-Path $ProjectRoot "configs\nam_baseline\data.json"
$ModelCfg = Join-Path $ProjectRoot "configs\nam_baseline\model.json"
$LearnCfg = Join-Path $ProjectRoot "configs\nam_baseline\learning.json"
$OutDir = Join-Path $ProjectRoot "outputs\nam_baseline"
$ModelOut = Join-Path $OutDir "model.nam"
$NamFullExe = Join-Path $ProjectRoot ".venv\Scripts\nam-full.exe"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $PrepareLog -Encoding UTF8
"" | Set-Content -LiteralPath $TrainingLog -Encoding UTF8

function Write-LogLine {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$Message
  )

  $Message | Tee-Object -FilePath $LogPath -Append
}

function Invoke-LoggedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][scriptblock]$Command,
    [Parameter(Mandatory = $true)][string]$FailureMessage
  )

  & $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
  $ExitCode = $LASTEXITCODE
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

  & cmd.exe /d /c $CommandText | Tee-Object -FilePath $LogPath -Append
  $ExitCode = $LASTEXITCODE
  if ($ExitCode -ne 0) {
    Write-LogLine -LogPath $LogPath -Message "ERROR: $FailureMessage (exit code $ExitCode)"
    exit 1
  }
}

if (-not (Test-Path $Python)) {
  throw ".venv not found. Please run scripts/setup_step1.ps1 first."
}

try {
  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"

  Write-LogLine -LogPath $PrepareLog -Message "=== Step2 Prepare NAM Baseline ==="
  Write-LogLine -LogPath $PrepareLog -Message "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine -LogPath $PrepareLog -Message "python executable: $Python"
  Write-LogLine -LogPath $PrepareLog -Message "project root: $ProjectRoot"
  Write-LogLine -LogPath $PrepareLog -Message "data config path: $DataCfg"
  Write-LogLine -LogPath $PrepareLog -Message "model config path: $ModelCfg"
  Write-LogLine -LogPath $PrepareLog -Message "learning config path: $LearnCfg"

  Write-LogLine -LogPath $TrainingLog -Message "=== Step2 NAM Training ==="
  Write-LogLine -LogPath $TrainingLog -Message "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine -LogPath $TrainingLog -Message "python executable: $Python"
  Write-LogLine -LogPath $TrainingLog -Message "pip executable: $Pip"
  Write-LogLine -LogPath $TrainingLog -Message "project root: $ProjectRoot"
  Write-LogLine -LogPath $TrainingLog -Message "PYTHONUTF8: $($env:PYTHONUTF8)"
  Write-LogLine -LogPath $TrainingLog -Message "PYTHONIOENCODING: $($env:PYTHONIOENCODING)"
  Write-LogLine -LogPath $TrainingLog -Message "data config path: $DataCfg"
  Write-LogLine -LogPath $TrainingLog -Message "model config path: $ModelCfg"
  Write-LogLine -LogPath $TrainingLog -Message "learning config path: $LearnCfg"
  Write-LogLine -LogPath $TrainingLog -Message "output directory: $OutDir"

  Write-LogLine -LogPath $TrainingLog -Message "[Step2] Checking neural-amp-modeler installation..."
  & $Python -c "import nam; print(getattr(nam, '__version__', 'unknown'))" 2>&1 | Tee-Object -FilePath $TrainingLog -Append
  if ($LASTEXITCODE -ne 0) {
    Write-LogLine -LogPath $TrainingLog -Message "[Step2] neural-amp-modeler not found, installing..."
    Invoke-LoggedCommand -LogPath $TrainingLog -Command { & $Pip install neural-amp-modeler } -FailureMessage "[Step2] Failed to install neural-amp-modeler"
  } else {
    Write-LogLine -LogPath $TrainingLog -Message "[Step2] neural-amp-modeler already installed, skip pip install."
  }

  Invoke-LoggedCommand -LogPath $TrainingLog -Command {
    & $Python ".\scripts\print_dependency_versions.py" torch torchaudio librosa matplotlib tensorboard soundfile scipy numpy pandas neural-amp-modeler
  } -FailureMessage "[Step2] Failed to capture dependency versions."

  Write-LogLine -LogPath $PrepareLog -Message "[Step2] Preparing baseline data/config..."
  Invoke-LoggedCommand -LogPath $PrepareLog -Command { & $Python ".\scripts\prepare_nam_baseline.py" } -FailureMessage "[Step2] Baseline preparation failed"

  if (-not $SkipTrain) {
    Write-LogLine -LogPath $TrainingLog -Message "[Step2] Starting NAM baseline training..."
    if (Test-Path $NamFullExe) {
      $TrainingCommand = ('chcp 65001>nul && "{0}" "{1}" "{2}" "{3}" "{4}" 2>&1' -f $NamFullExe, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
      Write-LogLine -LogPath $TrainingLog -Message "training command: $TrainingCommand"
      Invoke-CmdLoggedCommand -LogPath $TrainingLog -CommandText $TrainingCommand -FailureMessage "[Step2] Training failed"
    } else {
      Write-LogLine -LogPath $TrainingLog -Message "[Step2] nam-full.exe not found, fallback to module entrypoint nam.train.full"
      $TrainingCommand = ('chcp 65001>nul && "{0}" -m nam.train.full "{1}" "{2}" "{3}" "{4}" 2>&1' -f $Python, $DataCfg, $ModelCfg, $LearnCfg, $OutDir)
      Write-LogLine -LogPath $TrainingLog -Message "training command: $TrainingCommand"
      Invoke-CmdLoggedCommand -LogPath $TrainingLog -CommandText $TrainingCommand -FailureMessage "[Step2] Training failed"
    }
  } else {
    Write-LogLine -LogPath $TrainingLog -Message "[Step2] SkipTrain enabled. Reusing existing outputs for validation only."
  }

  $FoundNam = Get-ChildItem -Path $OutDir -Recurse -File -Filter "*.nam" `
    | Sort-Object LastWriteTime -Descending `
    | Select-Object -First 1

  if ($null -eq $FoundNam) {
    throw "[Step2] No .nam file found under: $OutDir"
  }

  if ($FoundNam.FullName -ne $ModelOut) {
    Copy-Item -LiteralPath $FoundNam.FullName -Destination $ModelOut -Force
  }

  Write-LogLine -LogPath $TrainingLog -Message "[Step2] Training finished. Check: $OutDir"
  Write-LogLine -LogPath $TrainingLog -Message "[Step2] source model file: $($FoundNam.FullName)"
  Write-LogLine -LogPath $TrainingLog -Message "[Step2] canonical model file: $ModelOut"
}
catch {
  Write-LogLine -LogPath $PrepareLog -Message "ERROR: $($_.Exception.Message)"
  Write-LogLine -LogPath $TrainingLog -Message "ERROR: $($_.Exception.Message)"
  exit 1
}
