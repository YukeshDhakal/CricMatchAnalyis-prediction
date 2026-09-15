# Auto-sync: commits and pushes any local changes in this repo so a second
# machine can pull the exact same working tree. Run on a schedule (see
# scripts/register_auto_sync_task.ps1); safe to run manually too.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot ".git"
$logFile = Join-Path $logDir "auto_sync.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logFile -Value "[$ts] $msg"
}

try {
    # NOTE: never redirect git's stderr with 2>&1 here -- in Windows PowerShell 5.1
    # that wraps git's normal stderr chatter (e.g. fetch/pull status lines) in
    # ErrorRecords and made this script report failure on successful runs.
    $branch = git rev-parse --abbrev-ref HEAD
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Not a git repo or git error: $branch"
        exit 1
    }

    git fetch origin $branch | Out-Null

    $status = git status --porcelain
    if (-not $status) {
        # Nothing local to commit; still make sure we're current with origin.
        git pull --rebase --autostash origin $branch | Out-Null
        exit 0
    }

    git add -A

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-sync: $ts" | Out-Null
    Write-Log "Committed local changes."

    git pull --rebase --autostash origin $branch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Rebase failed - resolve conflicts manually, then push."
        exit 1
    }

    git push origin $branch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Push failed."
        exit 1
    }

    Write-Log "Pushed to origin/$branch."
}
catch {
    Write-Log "Error: $_"
    exit 1
}
