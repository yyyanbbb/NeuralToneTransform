Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "outputs\nam_baseline"
$Canonical = Join-Path $OutDir "model.nam"

if (-not (Test-Path $OutDir)) {
  throw "[Step2-Finalize] Output directory does not exist: $OutDir"
}

$FoundNam = Get-ChildItem -Path $OutDir -Recurse -File -Filter "*.nam" `
  | Sort-Object LastWriteTime -Descending `
  | Select-Object -First 1

if ($null -eq $FoundNam) {
  Write-Host "[Step2-Finalize] No .nam found. Checking if training artifacts exist..."
  $Artifacts = Get-ChildItem -Path $OutDir -Recurse -File `
    | Where-Object { $_.Name -match "checkpoint|ckpt|config|events" } `
    | Select-Object -First 10

  if ($Artifacts.Count -gt 0) {
    Write-Host "[Step2-Finalize] Training artifacts found, but no exported .nam yet."
    $Artifacts | ForEach-Object { Write-Host "  - $($_.FullName)" }
    throw "[Step2-Finalize] Please run export step or rerun nam-full with export enabled."
  }

  throw "[Step2-Finalize] No .nam and no recognizable artifacts found under $OutDir"
}

if (-not (Test-Path $Canonical)) {
  Copy-Item -LiteralPath $FoundNam.FullName -Destination $Canonical -Force
}

$File = Get-Item -LiteralPath $Canonical
Write-Host "[Step2-Finalize] Success."
Write-Host "[Step2-Finalize] source model: $($FoundNam.FullName)"
Write-Host "[Step2-Finalize] canonical model: $($File.FullName)"
Write-Host "[Step2-Finalize] size(bytes): $($File.Length)"
Write-Host "[Step2-Finalize] last_write: $($File.LastWriteTime)"
