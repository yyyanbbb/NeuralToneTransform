Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "outputs\a2_baseline"
$Canonical = Join-Path $OutDir "model.nam"
$LogsDir = Join-Path $ProjectRoot "logs\a2"
$LogPath = Join-Path $LogsDir "finalize_a2_model.log"
$Python = Join-Path $ProjectRoot ".venv-a2\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $LogPath -Encoding UTF8

function Write-LogLine {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message)
  $Message | Tee-Object -FilePath $LogPath -Append
}

try {
  Write-LogLine "=== A2 Finalize Model ==="
  Write-LogLine "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
  Write-LogLine "project root: $ProjectRoot"
  Write-LogLine "output directory: $OutDir"
  Write-LogLine "canonical model path: $Canonical"

  if (-not (Test-Path $OutDir)) {
    throw "Output directory does not exist: $OutDir"
  }

  $FoundNam = Get-ChildItem -Path $OutDir -Recurse -File -Filter "*.nam" `
    | Sort-Object LastWriteTime -Descending `
    | Select-Object -First 1

  if ($null -eq $FoundNam) {
    throw "No .nam file found under $OutDir"
  }

  if ($FoundNam.FullName -ne $Canonical) {
    Copy-Item -LiteralPath $FoundNam.FullName -Destination $Canonical -Force
  }

  $File = Get-Item -LiteralPath $Canonical
  Write-LogLine "source model: $($FoundNam.FullName)"
  Write-LogLine "canonical model: $($File.FullName)"
  Write-LogLine "size(bytes): $($File.Length)"
  Write-LogLine "last_write: $($File.LastWriteTime)"

  if (-not (Test-Path $Python)) {
    throw ".venv-a2 Python not found: $Python"
  }

  $global:LASTEXITCODE = 0
  $InspectOutput = & $Python ".\scripts\a2\inspect_a2_model.py" 2>&1
  $InspectExitCode = $LASTEXITCODE
  $InspectOutput | ForEach-Object { Write-LogLine ([string]$_) }
  if ($InspectExitCode -ne 0) {
    throw "A2 model inspection did not pass"
  }
}
catch {
  Write-LogLine "ERROR: $($_.Exception.Message)"
  exit 1
}
