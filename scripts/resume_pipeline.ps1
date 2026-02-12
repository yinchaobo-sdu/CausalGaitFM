param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$OutputRoot = "outputs",
    [string]$ControlDir = "outputs/control",
    [string]$Profile = "local_4060_full",
    [string[]]$ExtraArgs = @()
)

$runOutputDir = Join-Path $OutputRoot $RunId
$statePath = Join-Path $runOutputDir "pipeline_state.json"
$runsDir = Join-Path $OutputRoot "runs"
$logPath = Join-Path $runsDir "$RunId.pipeline.resume.log"
$errPath = Join-Path $runsDir "$RunId.pipeline.resume.err.log"
$pidPath = Join-Path $runsDir "$RunId.pid"
$stopPath = Join-Path $ControlDir "$RunId.stop"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}

if (-not (Test-Path $statePath)) {
    Write-Warning "pipeline_state.json not found at $statePath; resume will still run and start fresh state if needed."
}

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
    "--resume",
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
    "--save-every-steps", "200"
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
Write-Host "resume_state=$statePath"
Write-Host "log=$logPath"
Write-Host "err_log=$errPath"
