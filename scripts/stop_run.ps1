param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$OutputRoot = "outputs",
    [string]$ControlDir = "outputs/control",
    [int]$TimeoutSeconds = 300,
    [switch]$Force
)

$runsDir = Join-Path $OutputRoot "runs"
$pidPath = Join-Path $runsDir "$RunId.pid"
$stopPath = Join-Path $ControlDir "$RunId.stop"

New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
New-Item -ItemType File -Force -Path $stopPath | Out-Null
Write-Host "stop_marker=$stopPath"

if (-not (Test-Path $pidPath)) {
    Write-Warning "PID file not found: $pidPath. Stop marker has been set."
    exit 0
}

$pidRaw = (Get-Content $pidPath -ErrorAction Stop | Select-Object -First 1).Trim()
if (-not [int]::TryParse($pidRaw, [ref]$null)) {
    Write-Warning "Invalid PID value in ${pidPath}: '$pidRaw'"
    exit 0
}
$pid = [int]$pidRaw

$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Write-Host "Process already exited: $pid"
    exit 0
}

Write-Host "waiting_for_pid=$pid timeout=${TimeoutSeconds}s"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $alive = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($null -eq $alive) {
        Write-Host "process_exited=$pid"
        exit 0
    }
    Start-Sleep -Seconds 2
}

if ($Force) {
    Write-Warning "Timeout reached. Force-killing PID $pid"
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    exit 0
}

Write-Warning "Timeout reached and process still running. Re-run with -Force if needed."
