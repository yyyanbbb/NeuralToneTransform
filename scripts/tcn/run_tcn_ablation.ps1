param(
  [switch]$SkipSmall,
  [switch]$SkipMedium,
  [switch]$SkipLarge,
  [switch]$ContinueOnError,
  [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Env:PYTHONIOENCODING = "utf-8"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = (Resolve-Path (Join-Path $RepoRoot ".venv-a2\Scripts\python.exe")).Path
$LogDir = Join-Path $RepoRoot "logs\tcn"
$ReportPath = Join-Path $RepoRoot "reports\CUSTOM_TCN_PROGRESS.md"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath) | Out-Null

function Write-FailureReport {
  param(
    [string]$Variant,
    [string]$Stage,
    [int]$ExitCode,
    [string]$LogPath
  )
  $relativeLog = Resolve-Path $LogPath -ErrorAction SilentlyContinue
  if ($relativeLog) {
    $relativeLog = [System.IO.Path]::GetRelativePath($RepoRoot, $relativeLog)
  } else {
    $relativeLog = $LogPath
  }
  Add-Content -Path $ReportPath -Value ""
  Add-Content -Path $ReportPath -Value "## Ablation Failure Log"
  Add-Content -Path $ReportPath -Value ""
  Add-Content -Path $ReportPath -Value "- $(Get-Date -Format o): $Variant $Stage failed with exit code $ExitCode. Log: ``$relativeLog``"
}

function Invoke-LoggedCommand {
  param(
    [string]$Variant,
    [string]$Stage,
    [string]$LogName,
    [string[]]$Arguments
  )
  $LogPath = Join-Path $LogDir $LogName
  "[$(Get-Date -Format o)] $Variant $Stage" | Out-File -FilePath $LogPath -Encoding utf8
  "Command: $Python $($Arguments -join ' ')" | Out-File -FilePath $LogPath -Encoding utf8 -Append
  $StdoutPath = "$LogPath.stdout.tmp"
  $StderrPath = "$LogPath.stderr.tmp"
  Remove-Item -Force -ErrorAction SilentlyContinue $StdoutPath, $StderrPath
  $Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath
  $StdoutText = Read-RedirectedText -Path $StdoutPath
  if ($StdoutText) {
    Write-Host $StdoutText
    Add-Content -Path $LogPath -Encoding UTF8 -Value $StdoutText
  }
  $StderrText = Read-RedirectedText -Path $StderrPath
  if ($StderrText) {
    Write-Host $StderrText
    Add-Content -Path $LogPath -Encoding UTF8 -Value $StderrText
  }
  Remove-Item -Force -ErrorAction SilentlyContinue $StdoutPath, $StderrPath
  $ExitCode = $Process.ExitCode
  if ($ExitCode -ne 0) {
    Write-FailureReport -Variant $Variant -Stage $Stage -ExitCode $ExitCode -LogPath $LogPath
    if (-not $ContinueOnError) {
      throw "$Variant $Stage failed with exit code $ExitCode. See $LogPath"
    }
  }
  return $ExitCode
}

function Read-RedirectedText {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return ""
  }
  $Bytes = [System.IO.File]::ReadAllBytes($Path)
  if ($Bytes.Length -eq 0) {
    return ""
  }
  if ($Bytes.Length -gt 1 -and ($Bytes[1] -eq 0 -or ($Bytes[0] -eq 255 -and $Bytes[1] -eq 254))) {
    return [System.Text.Encoding]::Unicode.GetString($Bytes).TrimEnd()
  }
  return [System.Text.Encoding]::UTF8.GetString($Bytes).TrimEnd()
}

$Variants = @(
  @{
    Name = "small"
    Skip = $SkipSmall
    ModelConfig = "configs/tcn_gated/small.json"
    FormalConfig = "configs/tcn_gated/training_formal_small.json"
    SmokeConfig = "configs/tcn_gated/training_smoke_small.json"
    OutputDir = "outputs/tcn_gated/small"
  },
  @{
    Name = "medium"
    Skip = $SkipMedium
    ModelConfig = "configs/tcn_gated/medium.json"
    FormalConfig = "configs/tcn_gated/training_formal_medium.json"
    SmokeConfig = "configs/tcn_gated/training_smoke_medium.json"
    OutputDir = "outputs/tcn_gated/medium"
  },
  @{
    Name = "large"
    Skip = $SkipLarge
    ModelConfig = "configs/tcn_gated/large.json"
    FormalConfig = "configs/tcn_gated/training_formal_large.json"
    SmokeConfig = "configs/tcn_gated/training_smoke_large.json"
    OutputDir = "outputs/tcn_gated/large"
  }
)

foreach ($Variant in $Variants) {
  if ($Variant.Skip) {
    Write-Host "Skipping $($Variant.Name)"
    continue
  }

  $TrainingConfig = if ($Smoke) { $Variant.SmokeConfig } else { $Variant.FormalConfig }
  $Name = $Variant.Name
  $Checkpoint = Join-Path $Variant.OutputDir "checkpoints/best.pt"
  $Prediction = Join-Path $Variant.OutputDir "prediction.wav"

  Invoke-LoggedCommand -Variant $Name -Stage "train" -LogName "$($Name)_train.log" -Arguments @(
    ".\src\ntt\tcn\train.py",
    "--model-config", $Variant.ModelConfig,
    "--training-config", $TrainingConfig
  ) | Out-Null

  Invoke-LoggedCommand -Variant $Name -Stage "infer" -LogName "$($Name)_infer.log" -Arguments @(
    ".\src\ntt\tcn\infer.py",
    "--checkpoint", $Checkpoint,
    "--input", "data/aligned/aligned_dry.wav",
    "--output", $Prediction
  ) | Out-Null

  Invoke-LoggedCommand -Variant $Name -Stage "verify" -LogName "$($Name)_verify.log" -Arguments @(
    ".\src\ntt\tcn\verify_tcn.py",
    "--output-dir", $Variant.OutputDir
  ) | Out-Null
}
