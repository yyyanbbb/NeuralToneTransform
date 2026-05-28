Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Failures = New-Object System.Collections.Generic.List[string]
$PassCount = 0
$FailCount = 0

function Add-Result {
  param(
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][string]$Message
  )

  if ($Passed) {
    $script:PassCount += 1
    Write-Host "[PASS] $Message"
  } else {
    $script:FailCount += 1
    $script:Failures.Add($Message)
    Write-Host "[FAIL] $Message"
  }
}

function Test-FileExists {
  param([Parameter(Mandatory = $true)][string]$Path)
  return Test-Path -LiteralPath $Path -PathType Leaf
}

Write-Host "=== NeuralToneTransform Reproducibility Verification ==="
Write-Host "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Host "project root: $ProjectRoot"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$DataConfig = Join-Path $ProjectRoot "configs\nam_baseline\data.json"
$ModelOutput = Join-Path $ProjectRoot "outputs\nam_baseline\model.nam"
$BaselineInput = Join-Path $ProjectRoot "data\raw\baseline_input.wav"
$BaselineOutput = Join-Path $ProjectRoot "data\raw\baseline_output.wav"
$RequiredLogs = @(
  (Join-Path $ProjectRoot "logs\step1_env_check.log"),
  (Join-Path $ProjectRoot "logs\step2_prepare_nam_baseline.log"),
  (Join-Path $ProjectRoot "logs\step2_nam_training.log"),
  (Join-Path $ProjectRoot "logs\step2_finalize_model.log")
)

Add-Result -Passed (Test-Path -LiteralPath (Join-Path $ProjectRoot ".venv") -PathType Container) -Message ".venv directory exists"
Add-Result -Passed (Test-FileExists $Requirements) -Message "requirements.txt exists"
Add-Result -Passed (Test-FileExists $VenvPython) -Message "virtualenv python exists at $VenvPython"

if (Test-FileExists $VenvPython) {
  & $VenvPython --version
  Add-Result -Passed ($LASTEXITCODE -eq 0) -Message "python executable runs successfully"

  & $VenvPython ".\scripts\print_dependency_versions.py" torch torchaudio librosa matplotlib tensorboard soundfile scipy numpy pandas neural-amp-modeler
  Add-Result -Passed ($LASTEXITCODE -eq 0) -Message "core baseline dependencies import successfully"
}

if (Test-FileExists $DataConfig) {
  $ConfigText = Get-Content -LiteralPath $DataConfig -Raw
  $HasPrivateAbsolutePath = $ConfigText -match '[A-Za-z]:\\\\Users\\\\' -or `
    $ConfigText -match '/Users/' -or `
    $ConfigText -match '/home/'
  Add-Result -Passed (-not $HasPrivateAbsolutePath) -Message "configs/nam_baseline/data.json does not contain private absolute paths"
} else {
  Add-Result -Passed $false -Message "configs/nam_baseline/data.json exists"
}

Add-Result -Passed (Test-FileExists $BaselineInput) -Message "baseline input WAV exists"
Add-Result -Passed (Test-FileExists $BaselineOutput) -Message "baseline output WAV exists"
Add-Result -Passed (Test-FileExists $ModelOutput) -Message "canonical NAM model exists at outputs/nam_baseline/model.nam"

foreach ($LogPath in $RequiredLogs) {
  Add-Result -Passed (Test-FileExists $LogPath) -Message "log exists: $LogPath"
}

Write-Host ""
Write-Host "=== Summary ==="
Write-Host "pass_count: $PassCount"
Write-Host "fail_count: $FailCount"

if ($FailCount -eq 0) {
  Write-Host "OVERALL: PASS"
  exit 0
}

Write-Host "OVERALL: FAIL"
foreach ($Failure in $Failures) {
  Write-Host " - $Failure"
}
exit 1
