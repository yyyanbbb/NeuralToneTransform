Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$LogsDir = Join-Path $ProjectRoot "logs\a2"
$LogPath = Join-Path $LogsDir "verify_a2_baseline.log"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $LogPath -Encoding UTF8

$PassCount = 0
$FailCount = 0
$Failures = New-Object System.Collections.Generic.List[string]

function Write-LogLine {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message)
  $Message | Tee-Object -FilePath $LogPath -Append
}

function Add-Result {
  param(
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][string]$Message
  )
  if ($Passed) {
    $script:PassCount += 1
    Write-LogLine "[PASS] $Message"
  } else {
    $script:FailCount += 1
    $script:Failures.Add($Message)
    Write-LogLine "[FAIL] $Message"
  }
}

function Test-FileExists {
  param([Parameter(Mandatory = $true)][string]$Path)
  return Test-Path -LiteralPath $Path -PathType Leaf
}

function Test-NoPrivateAbsolutePath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return $false
  }
  $Text = Get-Content -LiteralPath $Path -Raw
  return -not ($Text -match '[A-Za-z]:[\\/]+Users[\\/]+|/Users/|/home/|/mnt/c/Users/')
}

Write-LogLine "=== A2 Baseline Verification ==="
Write-LogLine "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-LogLine "project root: $ProjectRoot"

Add-Result -Passed (Test-Path -LiteralPath (Join-Path $ProjectRoot ".venv-a2") -PathType Container) -Message ".venv-a2 exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "requirements-a2.txt")) -Message "requirements-a2.txt exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a2_baseline\data.json")) -Message "A2 data config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a2_baseline\model_packed.json")) -Message "A2 model_packed config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a2_baseline\learning.json")) -Message "A2 learning config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "data\raw\baseline_input.wav")) -Message "baseline input WAV exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "data\raw\baseline_output.wav")) -Message "baseline output WAV exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "outputs\a2_baseline\model.nam")) -Message "outputs/a2_baseline/model.nam exists"
Add-Result -Passed (Test-Path -LiteralPath $LogsDir -PathType Container) -Message "logs/a2 exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "reports\a2_model_inspection.md")) -Message "A2 inspection report exists"

foreach ($Config in @("data.json", "model_packed.json", "learning.json")) {
  $ConfigPath = Join-Path $ProjectRoot "configs\a2_baseline\$Config"
  Add-Result -Passed (Test-NoPrivateAbsolutePath $ConfigPath) -Message "configs/a2_baseline/$Config has no private absolute path"
}

Write-LogLine ""
Write-LogLine "=== Summary ==="
Write-LogLine "pass_count: $PassCount"
Write-LogLine "fail_count: $FailCount"

if ($FailCount -eq 0) {
  Write-LogLine "OVERALL: PASS"
  exit 0
}

Write-LogLine "OVERALL: FAIL"
foreach ($Failure in $Failures) {
  Write-LogLine " - $Failure"
}
exit 1
