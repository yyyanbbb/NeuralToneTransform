param(
  [switch]$SkipTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $Python)) {
  throw ".venv not found. Please run scripts/setup_step1.ps1 first."
}

Write-Host "[Step2] Checking neural-amp-modeler installation..."
& $Python -c "import nam; print(getattr(nam, '__version__', 'unknown'))" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "[Step2] neural-amp-modeler not found, installing..."
  & $Pip install neural-amp-modeler
  if ($LASTEXITCODE -ne 0) {
    throw "[Step2] Failed to install neural-amp-modeler"
  }
} else {
  Write-Host "[Step2] neural-amp-modeler already installed, skip pip install."
}

Write-Host "[Step2] Preparing baseline data/config..."
& $Python ".\scripts\prepare_nam_baseline.py"

$DataCfg = Join-Path $ProjectRoot "configs\nam_baseline\data.json"
$ModelCfg = Join-Path $ProjectRoot "configs\nam_baseline\model.json"
$LearnCfg = Join-Path $ProjectRoot "configs\nam_baseline\learning.json"
$OutDir = Join-Path $ProjectRoot "outputs\nam_baseline"
$ModelOut = Join-Path $OutDir "model.nam"

if (-not $SkipTrain) {
  Write-Host "[Step2] Starting NAM baseline training..."
  $NamFullExe = Join-Path $ProjectRoot ".venv\Scripts\nam-full.exe"
  if (Test-Path $NamFullExe) {
    & $NamFullExe $DataCfg $ModelCfg $LearnCfg $OutDir
  } else {
    Write-Host "[Step2] nam-full.exe not found, fallback to module entrypoint nam.train.full"
    & $Python -m nam.train.full $DataCfg $ModelCfg $LearnCfg $OutDir
  }

  if ($LASTEXITCODE -ne 0) {
    throw "[Step2] Training failed with exit code $LASTEXITCODE"
  }
} else {
  Write-Host "[Step2] SkipTrain enabled. Reusing existing outputs for validation only."
}

$FoundNam = Get-ChildItem -Path $OutDir -Recurse -File -Filter "*.nam" `
  | Sort-Object LastWriteTime -Descending `
  | Select-Object -First 1

if ($null -eq $FoundNam) {
  throw "[Step2] No .nam file found under: $OutDir"
}

if (-not (Test-Path $ModelOut)) {
  Copy-Item -LiteralPath $FoundNam.FullName -Destination $ModelOut -Force
}

Write-Host "[Step2] Training finished. Check: $OutDir"
Write-Host "[Step2] source model file: $($FoundNam.FullName)"
Write-Host "[Step2] canonical model file: $ModelOut"
