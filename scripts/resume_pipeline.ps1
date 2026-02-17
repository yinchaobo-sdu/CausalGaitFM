[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$OutputRoot = "outputs",
    [string]$ControlDir = "outputs/control",
    [string]$Profile = "local_4060_full",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @(),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgsTail = @()
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repoRoot $PythonExe
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot $OutputRoot
}
if (-not [System.IO.Path]::IsPathRooted($ControlDir)) {
    $ControlDir = Join-Path $repoRoot $ControlDir
}

$runOutputDir = Join-Path $OutputRoot $RunId
$statePath = Join-Path $runOutputDir "pipeline_state.json"
$runsDir = Join-Path $OutputRoot "runs"
$logPath = Join-Path $runsDir "$RunId.pipeline.resume.log"
$errPath = Join-Path $runsDir "$RunId.pipeline.resume.err.log"
$pidPath = Join-Path $runsDir "$RunId.pid"
$stopPath = Join-Path $ControlDir "$RunId.stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

if (-not (Test-Path $statePath)) {
    Write-Warning "pipeline_state.json not found at $statePath; resume will still run and start fresh state if needed."
}

if ($ExtraArgsTail -and $ExtraArgsTail.Count -gt 0) {
    $ExtraArgs = @($ExtraArgs + $ExtraArgsTail)
}

if ($ExtraArgs.Count -eq 1) {
    $token = $ExtraArgs[0].Trim()
    if ($token.StartsWith("@(") -and $token.EndsWith(")")) {
        $matches = [regex]::Matches($token, '"([^"]*)"|''([^'']*)''')
        if ($matches.Count -gt 0) {
            $parsed = @()
            foreach ($m in $matches) {
                if ($m.Groups[1].Success) {
                    $parsed += $m.Groups[1].Value
                } elseif ($m.Groups[2].Success) {
                    $parsed += $m.Groups[2].Value
                }
            }
            if ($parsed.Count -gt 0) {
                $ExtraArgs = $parsed
            }
        }
    } elseif ($token.StartsWith("--") -and $token.Contains(",")) {
        $parts = $token.Split(",", 2)
        if ($parts.Count -eq 2) {
            $ExtraArgs = @($parts[0], $parts[1])
        }
    }
}

New-Item -ItemType Directory -Force -Path $runOutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
if (Test-Path $stopPath) {
    Remove-Item $stopPath -Force
}

if (Test-Path $pidPath) {
    $existingPid = (Get-Content $pidPath -Raw).Trim()
    if ($existingPid) {
        $existingProc = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($existingProc) {
            Write-Error "Run already active (run_id=$RunId, pid=$existingPid). Stop it first or use a new run_id."
            exit 1
        }
    }
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
    "--save-every-steps", "200",
    "--pipeline-autosave-sec", "3600"
) + $ExtraArgs

try {
    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $argList `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -ErrorAction Stop
} catch {
    Write-Error "Failed to start pipeline process: $($_.Exception.Message)"
    exit 1
}

if (-not $proc -or -not $proc.Id) {
    Write-Error "Start-Process returned no PID for run_id=$RunId."
    exit 1
}

Set-Content -Path $pidPath -Value $proc.Id

Write-Host "run_id=$RunId"
Write-Host "pid=$($proc.Id)"
Write-Host "resume_state=$statePath"
Write-Host "log=$logPath"
Write-Host "err_log=$errPath"
