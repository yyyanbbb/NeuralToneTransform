param(
  [switch]$SkipA1,
  [switch]$SkipA2,
  [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

$LogsDir = Join-Path $ProjectRoot "logs\inference"
$A1Log = Join-Path $LogsDir "a1_inference.log"
$A2Log = Join-Path $LogsDir "a2_inference.log"
$A1Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$A2Python = Join-Path $ProjectRoot ".venv-a2\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
"" | Set-Content -LiteralPath $A1Log -Encoding UTF8
"" | Set-Content -LiteralPath $A2Log -Encoding UTF8

function Invoke-InferenceStep {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )

  $CommandText = ('"{0}" {1}' -f $FilePath, ($Arguments -join " "))
  $CommandText | Tee-Object -FilePath $LogPath -Append
  $StdOut = Join-Path $LogsDir ([System.IO.Path]::GetRandomFileName())
  $StdErr = Join-Path $LogsDir ([System.IO.Path]::GetRandomFileName())
  $Process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr
  if (Test-Path $StdOut) {
    Get-Content -LiteralPath $StdOut | Tee-Object -FilePath $LogPath -Append
    Remove-Item -LiteralPath $StdOut -Force
  }
  if (Test-Path $StdErr) {
    Get-Content -LiteralPath $StdErr | Tee-Object -FilePath $LogPath -Append
    Remove-Item -LiteralPath $StdErr -Force
  }
  $ExitCode = $Process.ExitCode
  if ($ExitCode -ne 0) {
    "ERROR: inference step failed with exit code $ExitCode" | Tee-Object -FilePath $LogPath -Append
    if (-not $ContinueOnError) {
      exit $ExitCode
    }
  }
}

if (-not $SkipA1) {
  Invoke-InferenceStep -LogPath $A1Log -FilePath $A1Python -Arguments @(
    ".\scripts\inference\run_a1_inference.py",
    "--model", "outputs\a1_baseline\model.nam",
    "--input", "data\aligned\aligned_dry.wav",
    "--output", "outputs\a1_baseline\prediction.wav"
  )
}

if (-not $SkipA2) {
  Invoke-InferenceStep -LogPath $A2Log -FilePath $A2Python -Arguments @(
    ".\scripts\inference\run_a2_inference.py",
    "--model", "outputs\a2_baseline\model.nam",
    "--input", "data\aligned\aligned_dry.wav",
    "--lite-output", "outputs\a2_baseline\a2_lite_prediction.wav",
    "--full-output", "outputs\a2_baseline\a2_full_prediction.wav"
  )
}
