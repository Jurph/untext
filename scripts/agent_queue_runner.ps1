# Sequential LLM agent queue runner. Fires ONE task per invocation.
#
# Design constraints (per Jurph, 2026-07-02):
#   - Rate limits make LLM-enabled overseers unreliable, so THIS script contains
#     no LLM logic: it is a dumb cron-style dispatcher. State lives in the
#     filesystem; a dead run is recovered by timestamps, not by intelligence.
#   - Tasks run sequentially, one per 12-hour tick, to stay under token budgets.
#   - Every run logs wall-clock start/finish; task prompts also instruct the
#     agent to self-report day+time at the top and bottom of its output.
#
# Layout (relative to repo root):
#   .codex-tmp\agent-queue\tasks\pending\NNN-name.md   task prompt files; first
#       line "MODEL: sonnet" (or haiku/opus) selects the model; rest is prompt.
#   .codex-tmp\agent-queue\tasks\running\              at most one task at a time
#   .codex-tmp\agent-queue\tasks\done\                 completed prompts
#   .codex-tmp\agent-queue\tasks\failed\               gave up after MaxAttempts
#   .codex-tmp\agent-queue\results\NNN-name.md         agent stdout (the report)
#   .codex-tmp\agent-queue\results\NNN-name.attempts   attempt counter
#   .codex-tmp\agent-queue\runner.log                  timestamped dispatch log
#
# Register via zero_sweep_register.ps1 (schtasks, every 12 hours).

$ErrorActionPreference = "Stop"

$RepoRoot  = "C:\Users\Jurph\Documents\Python Scripts\untext"
$QueueRoot = Join-Path $RepoRoot ".codex-tmp\agent-queue"
$Pending   = Join-Path $QueueRoot "tasks\pending"
$Running   = Join-Path $QueueRoot "tasks\running"
$Done      = Join-Path $QueueRoot "tasks\done"
$Failed    = Join-Path $QueueRoot "tasks\failed"
$Results   = Join-Path $QueueRoot "results"
$LogFile   = Join-Path $QueueRoot "runner.log"
$ClaudeExe = "C:\Users\Jurph\.local\bin\claude.exe"
$MaxAttempts = 4
$StaleHours  = 11   # a running task older than this is presumed dead
$RunTimeoutMinutes = 45

function Log([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$stamp] $msg"
}

foreach ($dir in @($Pending, $Running, $Done, $Failed, $Results)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Set-Location $RepoRoot

# --- Single-flight: recover stale runs, otherwise defer to the live one ---
$runningTasks = Get-ChildItem $Running -Filter "*.md" -ErrorAction SilentlyContinue
foreach ($task in $runningTasks) {
    $ageHours = ((Get-Date) - $task.LastWriteTime).TotalHours
    if ($ageHours -gt $StaleHours) {
        Log "stale running task $($task.Name) (age ${ageHours}h) -> back to pending"
        Move-Item $task.FullName (Join-Path $Pending $task.Name) -Force
    } else {
        Log "task $($task.Name) still running (age ${ageHours}h); skipping this tick"
        exit 0
    }
}

# --- Pick the lowest-numbered pending task ---
$next = Get-ChildItem $Pending -Filter "*.md" | Sort-Object Name | Select-Object -First 1
if (-not $next) {
    Log "queue empty; nothing to do"
    exit 0
}

$taskName   = [System.IO.Path]::GetFileNameWithoutExtension($next.Name)
$attemptsFile = Join-Path $Results "$taskName.attempts"
$attempts = 0
if (Test-Path $attemptsFile) { $attempts = [int](Get-Content $attemptsFile) }
$attempts += 1
Set-Content -Path $attemptsFile -Value $attempts

if ($attempts -gt $MaxAttempts) {
    Log "task $taskName exceeded $MaxAttempts attempts -> failed"
    Move-Item $next.FullName (Join-Path $Failed $next.Name) -Force
    exit 0
}

# --- Parse optional leading headers (order-free, each on its own line) ---
#   MODEL: haiku|sonnet|opus   -> pass --model; omitted = CLI default ("Fable").
#   FLAGS: skip-permissions    -> pass --dangerously-skip-permissions (tasks
#                                 that must edit files; reviews stay read-only).
#   TIMEOUT: 90                -> per-task run timeout in minutes.
$lines = Get-Content $next.FullName
$model = $null
$skipPermissions = $false
foreach ($line in $lines | Select-Object -First 4) {
    if ($line -match "^MODEL:\s*(\S+)") { $model = $Matches[1] }
    elseif ($line -match "^FLAGS:.*skip-permissions") { $skipPermissions = $true }
    elseif ($line -match "^TIMEOUT:\s*(\d+)") { $RunTimeoutMinutes = [int]$Matches[1] }
    elseif ($line -notmatch "^\s*$") { break }
}
# Prompt body is NOT passed on the command line (quoting + length hazards);
# the agent is told to read the task file itself.

$runningPath = Join-Path $Running $next.Name
Move-Item $next.FullName $runningPath -Force
(Get-Item $runningPath).LastWriteTime = Get-Date   # staleness clock starts now

$resultFile = Join-Path $Results "$taskName.md"
$errFile    = Join-Path $Results "$taskName.stderr.log"
$modelLabel = if ($model) { $model } else { "cli-default" }
Log "dispatch $taskName model=$modelLabel attempt=$attempts"

# NOTE: no quotes inside the prompt -- Start-Process argument re-quoting eats
# embedded quotes. The relative path contains no spaces, so none are needed.
$bootstrap = "Read the file .codex-tmp/agent-queue/tasks/running/$($next.Name) and execute the instructions in it. Ignore any leading MODEL:/FLAGS:/TIMEOUT: header lines. Write your full report as your final response text."

# PS 5.1 Start-Process joins ArgumentList WITHOUT quoting, so wrap the prompt
# in explicit quotes (it contains none of its own).
$claudeArgs = @("-p", "`"$bootstrap`"")
if ($model) { $claudeArgs += @("--model", $model) }
if ($skipPermissions) { $claudeArgs += "--dangerously-skip-permissions" }

# --- Run headless; read-only review tasks need no permission bypass ---
$proc = Start-Process -FilePath $ClaudeExe `
    -ArgumentList $claudeArgs `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $resultFile `
    -RedirectStandardError $errFile `
    -WindowStyle Hidden -PassThru

$finished = $proc.WaitForExit($RunTimeoutMinutes * 60 * 1000)
if (-not $finished) {
    $proc.Kill()
    Log "task $taskName TIMED OUT after ${RunTimeoutMinutes}m -> back to pending (attempt $attempts)"
    Move-Item $runningPath (Join-Path $Pending $next.Name) -Force
    exit 0
}

# Success = finished in time AND a substantive report landed on stdout.
# ($proc.ExitCode via Start-Process -PassThru is unreliable after redirects.)
$resultBytes = 0
if (Test-Path $resultFile) { $resultBytes = (Get-Item $resultFile).Length }
if ($resultBytes -gt 200) {
    Log "task $taskName SUCCEEDED ($resultBytes bytes)"
    Move-Item $runningPath (Join-Path $Done $next.Name) -Force
} else {
    $errTail = ""
    if ((Test-Path $errFile) -and ((Get-Item $errFile).Length -gt 0)) {
        $errTail = (Get-Content $errFile -Tail 1) -join " "
    }
    Log "task $taskName FAILED (out=$resultBytes bytes; stderr: $errTail) -> back to pending (attempt $attempts of $MaxAttempts)"
    Move-Item $runningPath (Join-Path $Pending $next.Name) -Force
}
