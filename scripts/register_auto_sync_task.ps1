# Registers a Windows Scheduled Task that runs auto_sync.ps1 every 5 minutes,
# indefinitely, so this repo stays pushed to GitHub as you work locally.
# Run this once (as the current user) to install the task:
#   powershell -ExecutionPolicy Bypass -File scripts\register_auto_sync_task.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\auto_sync.ps1"
$taskName = "ThirdUmpire-AutoSync"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Auto-commits and pushes Third Umpire repo changes every 5 minutes." `
    -Force

Write-Host "Registered scheduled task '$taskName' (runs every 5 minutes)."
