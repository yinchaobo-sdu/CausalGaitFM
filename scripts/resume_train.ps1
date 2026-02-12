param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$OutputRoot = "outputs",
    [string]$ControlDir = "outputs/control",
    [string[]]$ExtraArgs = @()
)

$runOutputDir = Join-Path $OutputRoot $RunId
$checkpointPath = Join-Path $runOutputDir "last_model.pth"
$runsDir = Join-Path $OutputRoot "runs"
$logPath = Join-Path $runsDir "$RunId.resume.log"
$errPath = Join-Path $runsDir "$RunId.resume.err.log"
$pidPath = Join-Path $runsDir "$RunId.pid"
$stopPath = Join-Path $ControlDir "$RunId.stop"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}

if (-not (Test-Path $checkpointPath)) {
    Write-Error "Checkpoint not found: $checkpointPath"
    exit 1
}

New-Item -ItemType Directory -Force -Path $runOutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
if (Test-Path $stopPath) {
    Remove-Item $stopPath -Force
}

$argList = @(
    "-u",
    "-m", "project.train",
    "--device", "cuda",
    "--run-id", $RunId,
    "--control-dir", $ControlDir,
    "--output-dir", $runOutputDir,
    "--resume-from", $checkpointPath,
    "--auto-batch",
    "--min-batch-size", "4",
    "--use-amp", "true",
    "--amp-dtype", "fp16",
    "--allow-tf32", "true",
    "--cudnn-benchmark", "true",
    "--num-workers", "0",
    "--pin-memory", "true",
    "--persistent-workers", "false",
    "--prefetch-factor", "2"
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
Write-Host "resume_from=$checkpointPath"
Write-Host "log=$logPath"
Write-Host "err_log=$errPath"
