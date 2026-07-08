param(
    [string]$RunName = "",
    [int]$Lanes = 8,
    [int]$ChunkSize = 64,
    [int]$GeometryStart = 0,
    [int]$GeometryConfigs = 13824,
    [int]$GeometryStep = 16,
    [int]$IncludeLocalCleanup = 1,
    [int]$IncludeColorProposal = 1
)

$ErrorActionPreference = "Stop"

if ($Lanes -lt 1) {
    throw "Lanes must be at least 1"
}
if ($ChunkSize -lt 1) {
    throw "ChunkSize must be at least 1"
}
if ($GeometryConfigs -lt 0) {
    throw "GeometryConfigs must be non-negative"
}
if ($GeometryStart -lt 0) {
    throw "GeometryStart must be non-negative"
}
if ($GeometryStep -lt 1) {
    throw "GeometryStep must be at least 1"
}
if (($IncludeLocalCleanup -ne 0) -and ($IncludeLocalCleanup -ne 1)) {
    throw "IncludeLocalCleanup must be 0 or 1"
}
if (($IncludeColorProposal -ne 0) -and ($IncludeColorProposal -ne 1)) {
    throw "IncludeColorProposal must be 0 or 1"
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if ($RunName -eq "") {
    $RunName = "overnight-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$runDir = Join-Path $repo ("experiments\mask-grid\" + $RunName)
$shardDir = Join-Path $runDir "shards"
$logDir = Join-Path $runDir "logs"
$laneDir = Join-Path $runDir "lanes"
New-Item -ItemType Directory -Force -Path $shardDir, $logDir, $laneDir | Out-Null

$jobs = New-Object System.Collections.Generic.List[object]

function Add-GridJob {
    param(
        [string]$Preset,
        [int]$Start,
        [int]$Limit,
        [int]$Step,
        [string]$Name
    )
    $out = Join-Path $shardDir ($Name + ".jsonl")
    $log = Join-Path $logDir ($Name + ".log")
    $jobs.Add([pscustomobject]@{
        preset = $Preset
        start = $Start
        limit = $Limit
        step = $Step
        name = $Name
        out = $out
        log = $log
        expected_rows = $Limit * 17
    })
}

if ($IncludeColorProposal -eq 1) {
    Add-GridJob -Preset "color-proposal" -Start 0 -Limit 12 -Step 1 -Name "color-proposal-00000-00011"
}

if ($IncludeLocalCleanup -eq 1) {
    for ($start = 0; $start -lt 288; $start += 48) {
        $limit = [Math]::Min(48, 288 - $start)
        Add-GridJob -Preset "local-cleanup" -Start $start -Limit $limit -Step 1 -Name ("local-cleanup-{0:d5}-{1:d5}" -f $start, ($start + $limit - 1))
    }
}

for ($residue = 0; $residue -lt $Lanes; $residue++) {
    $start = $GeometryStart + $residue
    $limit = [Math]::Ceiling($GeometryConfigs / $Lanes)
    Add-GridJob -Preset "geometry-budget" -Start $start -Limit $limit -Step $GeometryStep -Name ("geometry-budget-start{0:d5}-step{1:d2}" -f $start, $GeometryStep)
}

$manifest = [pscustomobject]@{
    run_name = $RunName
    started_at = (Get-Date).ToString("o")
    repo = $repo
    lanes = $Lanes
    chunk_size = $ChunkSize
    geometry_start = $GeometryStart
    geometry_configs = $GeometryConfigs
    geometry_step = $GeometryStep
    fixture_count = 17
    run_dir = $runDir
    jobs = $jobs
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $runDir "manifest.json")

$laneJobs = @()
for ($lane = 0; $lane -lt $Lanes; $lane++) {
    $laneJobs += ,@()
}
for ($i = 0; $i -lt $jobs.Count; $i++) {
    $laneJobs[$i % $Lanes] += $jobs[$i]
}

$processes = @()
for ($lane = 0; $lane -lt $Lanes; $lane++) {
    $laneScript = Join-Path $laneDir ("lane-{0:d2}.ps1" -f $lane)
    $laneLog = Join-Path $logDir ("lane-{0:d2}.log" -f $lane)
    $laneJobs[$lane] | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $laneDir ("lane-{0:d2}.jobs.json" -f $lane))

    $script = @"
`$ErrorActionPreference = "Continue"
Set-Location "$repo"
`$jobs = Get-Content -Raw "$($laneDir)\lane-$("{0:d2}" -f $lane).jobs.json" | ConvertFrom-Json
foreach (`$job in `$jobs) {
    `$started = Get-Date
    "START `$(`$started.ToString('o')) `$(`$job.name) preset=`$(`$job.preset) start=`$(`$job.start) step=`$(`$job.step) limit=`$(`$job.limit)" | Tee-Object -FilePath "$laneLog" -Append
    uv run python scripts\run_mask_grid.py --preset `$job.preset --config-start `$job.start --config-step `$job.step --limit-configs `$job.limit --out `$job.out *> `$job.log
    `$exit = `$LASTEXITCODE
    `$rows = 0
    if (Test-Path `$job.out) {
        `$rows = (Get-Content `$job.out | Measure-Object -Line).Lines
    }
    `$finished = Get-Date
    "DONE `$(`$finished.ToString('o')) `$(`$job.name) exit=`$exit rows=`$rows expected=`$(`$job.expected_rows) elapsed_sec=`$([int](`$finished - `$started).TotalSeconds)" | Tee-Object -FilePath "$laneLog" -Append
}
"@
    $script | Set-Content -Encoding UTF8 $laneScript
    $quotedLaneScript = '"' + $laneScript + '"'
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $quotedLaneScript) -WindowStyle Hidden -PassThru
    $processes += [pscustomobject]@{
        lane = $lane
        pid = $proc.Id
        script = $laneScript
        log = $laneLog
        job_count = $laneJobs[$lane].Count
    }
}

$processes | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 (Join-Path $runDir "processes.json")

Write-Host "Started mask grid overnight run: $RunName"
Write-Host "Run directory: $runDir"
Write-Host "Processes:"
$processes | Format-Table -AutoSize
