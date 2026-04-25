param(
    [string]$VideoPath = "/dev/video0",
    [string]$Mode = "policy_rollout",   # shadow | policy_rollout
    [string]$ExactQ = "true",           # true | false
    [string]$BinaryPath = ".\main_shadow_mode.exe",
    [string]$OutputDir = ".\rollout_outputs",
    [switch]$StopOnError,
    [int]$PauseSecondsBetweenRuns = 3
)

$ErrorActionPreference = "Stop"

# -----------------------------
# Model list: edit here only
# -----------------------------
$Models = @(
    @{ Name = "bc";              Json = ".\bc_train.json" },
    @{ Name = "bc_dagger";       Json = ".\bc_dagger_policy_export.json" },
    @{ Name = "exact_q_policy";  Json = ".\dagger_exact_q_from_critic.json" },
    @{ Name = "proposed";        Json = ".\proposed_hybrid_policy_export.json" },
    @{ Name = "dagger_proposed"; Json = ".\hybrid_dagger_safe.json" }
)

function Test-RequiredFile {
    param([string]$PathToCheck)
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "Missing required file: $PathToCheck"
    }
}

function New-TimestampString {
    return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function New-RunCsvPath {
    param(
        [string]$BaseDir,
        [string]$ModelName,
        [string]$RunMode
    )
    $stamp = New-TimestampString
    $safeModel = $ModelName -replace "[^a-zA-Z0-9_-]", "_"
    $safeMode  = $RunMode -replace "[^a-zA-Z0-9_-]", "_"
    return (Join-Path $BaseDir ("rollout_{0}_{1}_{2}.csv" -f $safeModel, $safeMode, $stamp))
}

function Write-RunBanner {
    param(
        [int]$Index,
        [int]$Total,
        [string]$ModelName,
        [string]$JsonPath,
        [string]$CsvPath,
        [string]$RunMode,
        [string]$ExactQFlag
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ("[{0}/{1}] MODEL: {2}" -f $Index, $Total, $ModelName) -ForegroundColor Cyan
    Write-Host ("  JSON   : {0}" -f $JsonPath)
    Write-Host ("  CSV    : {0}" -f $CsvPath)
    Write-Host ("  MODE   : {0}" -f $RunMode)
    Write-Host ("  EXACTQ : {0}" -f $ExactQFlag)
    Write-Host "============================================================" -ForegroundColor Cyan
}

# -----------------------------
# Preflight checks
# -----------------------------
Test-RequiredFile $BinaryPath

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

foreach ($m in $Models) {
    Test-RequiredFile $m.Json
}

$summaryPath = Join-Path $OutputDir ("run_summary_{0}.csv" -f (New-TimestampString))
$summaryRows = @()

Write-Host "Binary    : $BinaryPath" -ForegroundColor Green
Write-Host "VideoPath : $VideoPath" -ForegroundColor Green
Write-Host "Mode      : $Mode" -ForegroundColor Green
Write-Host "ExactQ    : $ExactQ" -ForegroundColor Green
Write-Host "OutputDir : $OutputDir" -ForegroundColor Green
Write-Host "Summary   : $summaryPath" -ForegroundColor Green

# -----------------------------
# Main loop
# -----------------------------
$total = $Models.Count
$index = 0

foreach ($m in $Models) {
    $index += 1
    $csvPath = New-RunCsvPath -BaseDir $OutputDir -ModelName $m.Name -RunMode $Mode

    Write-RunBanner -Index $index -Total $total -ModelName $m.Name -JsonPath $m.Json -CsvPath $csvPath -RunMode $Mode -ExactQFlag $ExactQ

    $startTime = Get-Date
    $status = "OK"
    $exitCode = $null
    $errorText = ""

    try {
        $argList = @(
            $VideoPath,
            $m.Json,
            $m.Name,
            $csvPath,
            $Mode,
            $ExactQ
        )

        Write-Host ("Launching: {0} {1}" -f $BinaryPath, ($argList -join " ")) -ForegroundColor Yellow
        & $BinaryPath @argList
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            $status = "FAILED"
            $errorText = "Process exited with code $exitCode"
            Write-Host $errorText -ForegroundColor Red
            if ($StopOnError) {
                throw $errorText
            }
        }
        else {
            Write-Host "Run completed successfully." -ForegroundColor Green
        }
    }
    catch {
        $status = "FAILED"
        $exitCode = if ($null -eq $exitCode) { -1 } else { $exitCode }
        $errorText = $_.Exception.Message
        Write-Host ("Run failed: {0}" -f $errorText) -ForegroundColor Red
        if ($StopOnError) {
            $endTime = Get-Date
            $summaryRows += [pscustomobject]@{
                model_name = $m.Name
                json_path = (Resolve-Path $m.Json).Path
                csv_path = $csvPath
                mode = $Mode
                exact_q = $ExactQ
                start_time = $startTime.ToString("s")
                end_time = $endTime.ToString("s")
                status = $status
                exit_code = $exitCode
                error = $errorText
            }
            $summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryPath
            throw
        }
    }
    finally {
        $endTime = Get-Date
        $resolvedJson = try { (Resolve-Path $m.Json).Path } catch { $m.Json }
        $summaryRows += [pscustomobject]@{
            model_name = $m.Name
            json_path = $resolvedJson
            csv_path = $csvPath
            mode = $Mode
            exact_q = $ExactQ
            start_time = $startTime.ToString("s")
            end_time = $endTime.ToString("s")
            status = $status
            exit_code = $exitCode
            error = $errorText
        }
        $summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryPath
    }

    if ($index -lt $total -and $PauseSecondsBetweenRuns -gt 0) {
        Write-Host ("Pausing {0}s before next model..." -f $PauseSecondsBetweenRuns) -ForegroundColor DarkGray
        Start-Sleep -Seconds $PauseSecondsBetweenRuns
    }
}

Write-Host "" 
Write-Host "All runs finished." -ForegroundColor Green
Write-Host ("Summary saved to: {0}" -f $summaryPath) -ForegroundColor Green
Write-Host "" 
$summaryRows | Format-Table -AutoSize
