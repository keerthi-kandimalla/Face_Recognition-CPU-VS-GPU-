$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'venv\Scripts\python.exe'
$taskName = 'FaceRecognitionAutoGitSync'

if (-not (Test-Path $python)) {
    throw "Project Python interpreter not found: $python"
}

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    $gitCandidate = Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe'
    if (Test-Path $gitCandidate) {
        $env:Path = (Split-Path -Parent $gitCandidate) + ';' + $env:Path
        $git = Get-Command git -ErrorAction SilentlyContinue
    }
}
if (-not $git) {
    throw 'Git was not found. Install Git before enabling automatic sync.'
}

$remote = & git -C $root remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    throw "Configure the GitHub remote named 'origin' before enabling automatic sync."
}

$gitRoot = Join-Path $env:LOCALAPPDATA 'Programs\Git'
$credentialManager = Get-ChildItem $gitRoot -Filter 'git-credential-manager.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $credentialManager) {
    throw 'Git Credential Manager was not found; HTTPS auto-push cannot be authenticated safely.'
}
& git -C $root config --local credential.helper $credentialManager
if ($LASTEXITCODE -ne 0) {
    throw 'Could not configure Git Credential Manager for this repository.'
}

$scriptPath = Join-Path $PSScriptRoot 'auto_git_sync.py'
$userId = (& whoami.exe).Trim()
$action = New-ScheduledTaskAction -Execute $python -Argument "`"$scriptPath`" watch" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "Installed and started '$taskName'. The watcher log is at logs\auto_git_sync.log."