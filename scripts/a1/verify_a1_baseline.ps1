Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$LogsDir = Join-Path $ProjectRoot "logs\a1"
$LogPath = Join-Path $LogsDir "verify_a1_baseline.log"
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

Write-LogLine "=== A1 Baseline Verification ==="
Write-LogLine "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-LogLine "project root: $ProjectRoot"

Add-Result -Passed (Test-Path -LiteralPath (Join-Path $ProjectRoot ".venv") -PathType Container) -Message ".venv exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "requirements-a1.txt")) -Message "requirements-a1.txt exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a1_baseline\data.json")) -Message "A1 data config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a1_baseline\model.json")) -Message "A1 model config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "configs\a1_baseline\learning.json")) -Message "A1 learning config exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "data\raw\baseline_input.wav")) -Message "baseline input WAV exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "data\raw\baseline_output.wav")) -Message "baseline output WAV exists"
Add-Result -Passed (Test-FileExists (Join-Path $ProjectRoot "outputs\a1_baseline\model.nam")) -Message "outputs/a1_baseline/model.nam exists"
Add-Result -Passed (Test-Path -LiteralPath $LogsDir -PathType Container) -Message "logs/a1 exists"

foreach ($Config in @("data.json", "model.json", "learning.json")) {
  $ConfigPath = Join-Path $ProjectRoot "configs\a1_baseline\$Config"
  Add-Result -Passed (Test-NoPrivateAbsolutePath $ConfigPath) -Message "configs/a1_baseline/$Config has no private absolute path"
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
