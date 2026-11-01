# Install ELAgent and expose the `ela` command on the user PATH.
# Works with the `python` that comes first on PATH (3.10+).
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyVersion = python -c "import sys; print(sys.version.split()[0])"
$pyExe = python -c "import sys; print(sys.executable)"
Write-Host "Python $pyVersion @ $pyExe" -ForegroundColor DarkGray
Write-Host "Installing ELAgent from $repo ..." -ForegroundColor Cyan

# setuptools is needed for the editable build; install it first so that
# --no-build-isolation can work offline.
python -m pip install --user --quiet --upgrade setuptools
python -m pip install --user -e $repo --no-build-isolation
if ($LASTEXITCODE -ne 0) {
    Write-Host "Retrying with build isolation ..." -ForegroundColor Yellow
    python -m pip install --user -e $repo
}
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

$scripts = (python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))").Trim()

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
$parts = $userPath -split ';' | Where-Object { $_ -ne "" }

if ($parts -notcontains $scripts) {
    $newPath = if ($userPath.TrimEnd(';')) { "$($userPath.TrimEnd(';'));$scripts" } else { $scripts }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added to user PATH: $scripts" -ForegroundColor Yellow
} else {
    Write-Host "PATH already contains: $scripts" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. Open a NEW terminal, then run:" -ForegroundColor Green
Write-Host "  ela                 # TUI" -ForegroundColor White
Write-Host "  ela --doctor        # endpoint / model / prompt budget" -ForegroundColor White
