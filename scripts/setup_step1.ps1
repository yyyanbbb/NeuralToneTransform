Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "[Step1] Project root: $ProjectRoot"

if (-not (Test-Path ".venv")) {
  Write-Host "[Step1] Creating virtual environment..."
  python -m venv .venv
} else {
  Write-Host "[Step1] .venv already exists, reusing."
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Python executable not found in .venv: $Python"
}

Write-Host "[Step1] Upgrading pip..."
& $Python -m pip install --upgrade pip

Write-Host "[Step1] Installing dependencies..."
& $Python -m pip install torch torchaudio librosa matplotlib tensorboard soundfile scipy numpy pandas

$dirs = @(
  "data\raw",
  "data\aligned",
  "data\chunks\train",
  "data\chunks\val",
  "data\chunks\test",
  "src",
  "runs",
  "checkpoints",
  "reports",
  "configs\nam_baseline",
  "outputs\nam_baseline",
  "scripts"
)

foreach ($d in $dirs) {
  New-Item -ItemType Directory -Path $d -Force | Out-Null
}

Write-Host "[Step1] Running environment checks..."
& $Python ".\scripts\check_env.py"

Write-Host "[Step1] Done."
