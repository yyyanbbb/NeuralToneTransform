Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "outputs\nam_baseline"
$Canonical = Join-Path $OutDir "model.nam"
$LogsDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogsDir "step2_finalize_model.log"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $LogPath -Encoding UTF8

function Write-LogLine {
  param([Parameter(Mandatory = $true)][string]$Message)
  $Message | Tee-Object -FilePath $LogPath -Append
}

try {
  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"

  Write-LogLine "=== Step2 Finalize Model ==="
  Write-LogLine "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine "project root: $ProjectRoot"
  Write-LogLine "output directory: $OutDir"
  Write-LogLine "canonical model path: $Canonical"

  if (Test-Path $Python) {
    Write-LogLine "python executable: $Python"
    Write-LogLine "PYTHONUTF8: $($env:PYTHONUTF8)"
    Write-LogLine "PYTHONIOENCODING: $($env:PYTHONIOENCODING)"
    & $Python ".\scripts\print_dependency_versions.py" neural-amp-modeler torch 2>&1 | Tee-Object -FilePath $LogPath -Append
  }

  if (-not (Test-Path $OutDir)) {
    throw "[Step2-Finalize] Output directory does not exist: $OutDir"
  }

  $FoundNam = Get-ChildItem -Path $OutDir -Recurse -File -Filter "*.nam" `
    | Sort-Object LastWriteTime -Descending `
    | Select-Object -First 1

  if ($null -eq $FoundNam) {
    Write-LogLine "[Step2-Finalize] No .nam found. Checking if training artifacts exist..."
    $Artifacts = Get-ChildItem -Path $OutDir -Recurse -File `
      | Where-Object { $_.Name -match "checkpoint|ckpt|config|events" } `
      | Select-Object -First 10

    if ($Artifacts.Count -gt 0) {
      Write-LogLine "[Step2-Finalize] Training artifacts found, but no exported .nam yet."
      $Artifacts | ForEach-Object { Write-LogLine "  - $($_.FullName)" }
      throw "[Step2-Finalize] Please run export step or rerun nam-full with export enabled."
    }

    throw "[Step2-Finalize] No .nam and no recognizable artifacts found under $OutDir"
  }

  if ($FoundNam.FullName -ne $Canonical) {
    Copy-Item -LiteralPath $FoundNam.FullName -Destination $Canonical -Force
  }

  $File = Get-Item -LiteralPath $Canonical
  Write-LogLine "[Step2-Finalize] Success."
  Write-LogLine "[Step2-Finalize] source model: $($FoundNam.FullName)"
  Write-LogLine "[Step2-Finalize] canonical model: $($File.FullName)"
  Write-LogLine "[Step2-Finalize] size(bytes): $($File.Length)"
  Write-LogLine "[Step2-Finalize] last_write: $($File.LastWriteTime)"
}
catch {
  Write-LogLine "ERROR: $($_.Exception.Message)"
  exit 1
}
