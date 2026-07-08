param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [int]$TopN = 20,
    [switch]$RunTelea
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path $RunDir)) {
    throw "RunDir does not exist: $RunDir"
}

$manifestPath = Join-Path $RunDir "manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Missing manifest: $manifestPath"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$audit = foreach ($job in $manifest.jobs) {
    $rows = if (Test-Path $job.out) { (Get-Content $job.out | Measure-Object -Line).Lines } else { 0 }
    [pscustomobject]@{
        name = $job.name
        preset = $job.preset
        start = $job.start
        limit = $job.limit
        rows = $rows
        expected = $job.expected_rows
        complete = ($rows -eq $job.expected_rows)
        out = $job.out
    }
}

$auditPath = Join-Path $RunDir "shard-audit.csv"
$audit | Sort-Object complete, preset, start | Export-Csv -NoTypeInformation -Encoding UTF8 $auditPath

$incomplete = @($audit | Where-Object { -not $_.complete })
if ($incomplete.Count -gt 0) {
    Write-Host "Incomplete shards found. Wrote audit: $auditPath"
    $incomplete | Format-Table -AutoSize
    throw "Refusing to summarize until all expected rows are present."
}

$presets = $audit | Select-Object -ExpandProperty preset -Unique
foreach ($preset in $presets) {
    $combinedName = if ($preset -eq "geometry-budget") { "geometry-budget-strided" } else { $preset }
    $combinedJsonl = Join-Path $RunDir ($combinedName + ".jsonl")
    $csv = Join-Path $RunDir ($combinedName + ".csv")
    $topJson = Join-Path $RunDir ($combinedName + "-top.json")
    $files = @($audit | Where-Object { $_.preset -eq $preset } | Sort-Object start | Select-Object -ExpandProperty out)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath (Split-Path -Parent $combinedJsonl)).Path + "\" + (Split-Path -Leaf $combinedJsonl), [string[]](Get-Content $files), $utf8NoBom)
    uv run python scripts\summarize_mask_grid.py $combinedJsonl --out $csv --top-json $topJson --top-n $TopN
    if ($RunTelea -and (Test-Path $topJson) -and ((Get-Item $topJson).Length -gt 2)) {
        $telea = Join-Path $RunDir ($combinedName + "-top-telea.jsonl")
        uv run python scripts\run_inpaint_eval.py --configs $topJson --method telea --out $telea
    }
}

Write-Host "Collected and summarized run: $RunDir"
Write-Host "Audit: $auditPath"
