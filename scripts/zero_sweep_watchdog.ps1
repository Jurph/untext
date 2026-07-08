# Watchdog for the zero-corpus 2.5% detector FP-baseline sweep.
#
# Token-proof by design: pure PowerShell + local python, no LLM calls anywhere.
# Intended to run every 15 minutes via schtasks (see zero_sweep_register.ps1).
# Each tick:
#   1. If a sweep process is already running -> log progress, exit.
#   2. If all 415 records exist -> run post-processing once (summary if missing,
#      post-hoc overlays, offline threshold-replay grid), write DONE marker, exit.
#   3. Otherwise relaunch the chunked supervisor with --resume (records are
#      per-image, so nothing is recomputed).
#
# Logs: append-only, timestamped, at .codex-tmp\zero_sweep_watchdog.log

$ErrorActionPreference = "Stop"

$RepoRoot   = "C:\Users\Jurph\Documents\Python Scripts\untext"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$OutRoot    = Join-Path $RepoRoot "tests\images\zero_detector_fp_baseline"
$RecordsDir = Join-Path $OutRoot "threshold_025\records"
$LogFile    = Join-Path $RepoRoot ".codex-tmp\zero_sweep_watchdog.log"
$DoneMarker = Join-Path $OutRoot "SWEEP_COMPLETE.marker"
$ExpectedRecords = 415

function Log([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$stamp] $msg"
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
Set-Location $RepoRoot   # schtasks starts tasks in System32; paths below are repo-relative

if (Test-Path $DoneMarker) {
    exit 0   # All done. Task can be deleted at leisure; stay silent to keep the log clean.
}

$recordCount = 0
if (Test-Path $RecordsDir) {
    $recordCount = (Get-ChildItem $RecordsDir -Filter "*.json" | Measure-Object).Count
}

# Is any sweep process (supervisor or chunk worker) still alive?
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "run_has_text_2_detector_threshold_sweep" }

if ($alive) {
    $alivePids = ($alive | ForEach-Object { $_.ProcessId }) -join ","
    Log "alive pids=$alivePids records=$recordCount/$ExpectedRecords"
    exit 0
}

if ($recordCount -lt $ExpectedRecords) {
    Log "relaunch records=$recordCount/$ExpectedRecords"
    $sweepArgs = @(
        "tests\run_has_text_2_detector_threshold_sweep.py",
        "tests\images\zero",
        "--out-root", "tests\images\zero_detector_fp_baseline",
        "--thresholds", "0.025",
        "--chunk-size", "20",
        "--max-retries", "1",
        "--resume"
    )
    # Detached: the supervisor runs chunked workers until completion or its own death;
    # the next watchdog tick picks up whatever state remains.
    Start-Process -FilePath $VenvPython -ArgumentList $sweepArgs -WorkingDirectory $RepoRoot -WindowStyle Hidden
    Log "relaunched supervisor"
    exit 0
}

# --- All records present: one-time post-processing ---
Log "records complete ($recordCount). post-processing"

$SummaryJson = Join-Path $OutRoot "threshold_025\summary.json"
if (-not (Test-Path $SummaryJson)) {
    # Final supervisor pass resumes all records, writes summaries, then exits.
    & $VenvPython "tests\run_has_text_2_detector_threshold_sweep.py" "tests\images\zero" `
        --out-root "tests\images\zero_detector_fp_baseline" --thresholds 0.025 `
        --chunk-size 20 --max-retries 1 --resume 2>&1 |
        Add-Content -Path $LogFile
    Log "summary pass exit=$LASTEXITCODE"
}

& $VenvPython ".codex-tmp\make_overlays.py" "tests\images\zero_detector_fp_baseline\threshold_025" "tests\images\zero" 2>&1 |
    Add-Content -Path $LogFile
Log "overlays exit=$LASTEXITCODE"

& $VenvPython "tests\analyze_has_text_2_detector_threshold_sweep.py" "tests\images\zero_detector_fp_baseline" `
    --input-dir "tests\images\zero" --source-threshold 0.025 `
    --thresholds 0.30 0.15 0.10 0.05 0.025 2>&1 |
    Add-Content -Path $LogFile
Log "replay grid exit=$LASTEXITCODE"

& $VenvPython ".codex-tmp\build_zero_review.py" "tests\images\zero_detector_fp_baseline\threshold_025" 2>&1 |
    Add-Content -Path $LogFile
Log "review csv exit=$LASTEXITCODE"

Set-Content -Path $DoneMarker -Value ("completed " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Log "DONE - marker written"
