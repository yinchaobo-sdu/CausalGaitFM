[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$RunId = "",
    [string]$OutputRoot = "outputs",
    [string]$ControlDir = "outputs/control",
    [string]$Profile = "local_4060_full",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}

$runOutputDir = Join-Path $OutputRoot $RunId
$runsDir = Join-Path $OutputRoot "runs"
$logPath = Join-Path $runsDir "$RunId.pipeline.log"
$errPath = Join-Path $runsDir "$RunId.pipeline.err.log"
$pidPath = Join-Path $runsDir "$RunId.pid"
$stopPath = Join-Path $ControlDir "$RunId.stop"

New-Item -ItemType Directory -Force -Path $runOutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
if (Test-Path $stopPath) {
    Remove-Item $stopPath -Force
}

$argList = @(
    "-u",
    "-m", "project.pipeline",
    "--profile", $Profile,
    "--device", "cuda",
    "--run-id", $RunId,
    "--control-dir", $ControlDir,
    "--output-dir", $runOutputDir,
    "--auto-batch",
    "--min-batch-size", "4",
    "--use-amp", "true",
    "--amp-dtype", "fp16",
    "--allow-tf32", "true",
    "--cudnn-benchmark", "true",
    "--num-workers", "0",
    "--pin-memory", "true",
    "--persistent-workers", "false",
    "--prefetch-factor", "2",
    "--check-stop-every", "20",
    "--save-every-steps", "200",
    "--pipeline-autosave-sec", "3600"
) + $ExtraArgs

$proc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $argList `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errPath

Set-Content -Path $pidPath -Value $proc.Id

Write-Host "run_id=$RunId"
Write-Host "pid=$($proc.Id)"
Write-Host "output_dir=$runOutputDir"
Write-Host "log=$logPath"
Write-Host "err_log=$errPath"
