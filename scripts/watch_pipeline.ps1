param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$OutputRoot = "outputs",
    [int]$RefreshSec = 2,
    [switch]$ShowLog,
    [int]$LogLines = 20
)

function Format-Duration {
    param([Nullable[int]]$Seconds)
    if ($null -eq $Seconds) { return "N/A" }
    if ($Seconds -lt 0) { $Seconds = 0 }
    $ts = [TimeSpan]::FromSeconds($Seconds)
    return ("{0:00}:{1:00}:{2:00}" -f [int]$ts.TotalHours, $ts.Minutes, $ts.Seconds)
}

function Read-StateSafely {
    param([string]$Path)
    try {
        if (-not (Test-Path $Path)) {
            return $null
        }
        return Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

$runOutputDir = Join-Path $OutputRoot $RunId
$statePath = Join-Path $runOutputDir "pipeline_state.json"
$primaryLogPath = Join-Path (Join-Path $OutputRoot "runs") "$RunId.pipeline.log"
$resumeLogPath = Join-Path (Join-Path $OutputRoot "runs") "$RunId.pipeline.resume.log"

Write-Host "watching run_id=$RunId"
Write-Host "state_file=$statePath"
Write-Host "refresh=${RefreshSec}s"

while ($true) {
    $state = Read-StateSafely -Path $statePath
    Clear-Host
    Write-Host "Pipeline Monitor"
    Write-Host ("run_id: {0}" -f $RunId)
    Write-Host ("state : {0}" -f $statePath)
    Write-Host ""

    if ($null -eq $state) {
        Write-Host "status : waiting_for_state_file"
        Start-Sleep -Seconds $RefreshSec
        continue
    }

    $status = [string]($state.status)
    $progress = if ($null -eq $state.progress_percent) { 0.0 } else { [double]$state.progress_percent }
    $currentStageName = ""
    $currentStageIndex = 0
    $stageTotal = if ($null -eq $state.stages_total) { 10 } else { [int]$state.stages_total }
    if ($null -ne $state.current_stage) {
        $currentStageName = [string]($state.current_stage.name)
        $currentStageIndex = [int]($state.current_stage.index)
    }

    $subName = ""
    $subDone = 0
    $subTotal = 1
    if ($null -ne $state.current_substage) {
        $subName = [string]($state.current_substage.name)
        $subDone = [int]($state.current_substage.done)
        $subTotal = [int]($state.current_substage.total)
        if ($subTotal -le 0) { $subTotal = 1 }
    }

    $startedAt = $null
    if ($null -ne $state.started_at) { $startedAt = [int]$state.started_at }
    $nowUnix = [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $elapsedSec = $null
    if ($null -ne $startedAt) { $elapsedSec = [Math]::Max(0, $nowUnix - $startedAt) }
    $etaSec = $null
    if ($null -ne $state.eta_seconds) { $etaSec = [int]$state.eta_seconds }

    Write-Host ("status       : {0}" -f $status)
    Write-Host ("progress     : {0:N2}%" -f $progress)
    Write-Host ("stage        : {0}/{1}  {2}" -f $currentStageIndex, $stageTotal, $currentStageName)
    Write-Host ("substage     : {0}/{1}  {2}" -f $subDone, $subTotal, $subName)
    Write-Host ("elapsed      : {0}" -f (Format-Duration -Seconds $elapsedSec))
    Write-Host ("eta          : {0}" -f (Format-Duration -Seconds $etaSec))

    if ($null -ne $state.last_error -and [string]$state.last_error -ne "") {
        Write-Host ("last_error   : {0}" -f [string]$state.last_error)
    }

    if ($ShowLog) {
        $logCandidates = @()
        if (Test-Path $primaryLogPath) {
            $logCandidates += Get-Item $primaryLogPath
        }
        if (Test-Path $resumeLogPath) {
            $logCandidates += Get-Item $resumeLogPath
        }

        $logPath = $null
        if ($logCandidates.Count -gt 0) {
            $logPath = ($logCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
        }
        Write-Host ""
        Write-Host ("log_tail ({0})" -f $logPath)
        if ($null -ne $logPath -and (Test-Path $logPath)) {
            Get-Content $logPath -Tail $LogLines
        } else {
            Write-Host "log_not_found"
        }
    }

    if ($status -in @("completed", "stopped", "failed")) {
        Write-Host ""
        Write-Host ("terminal_status={0}, exiting monitor." -f $status)
        break
    }

    Start-Sleep -Seconds $RefreshSec
}
