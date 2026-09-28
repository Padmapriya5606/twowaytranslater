# PowerShell Startup Script for ISL Two-Way Translator
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Starting ISL Two-Way Translator Web Application" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Detect Python interpreter from virtual environment if available
$VenvPy = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    Write-Host "[1/2] Using Virtual Environment: $VenvPy" -ForegroundColor Yellow
    $PyCmd = $VenvPy
} else {
    Write-Host "[1/2] Using System Python..." -ForegroundColor Yellow
    $PyCmd = "python"
}

Write-Host "[2/2] Launching server on http://localhost:8000 ..." -ForegroundColor Yellow
& $PyCmd run_app.py
