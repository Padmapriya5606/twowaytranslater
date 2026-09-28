@echo off
title ISL Two-Way Translator Server
echo =================================================================
echo   Starting ISL Two-Way Translator Web Application
echo =================================================================
echo.
cd /d "%~dp0"

REM Detect virtual environment python
if exist "venv\Scripts\python.exe" (
    echo [1/2] Using Virtual Environment: venv\Scripts\python.exe
    set "PY_CMD=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    echo [1/2] Using Virtual Environment: ..\venv\Scripts\python.exe
    set "PY_CMD=..\venv\Scripts\python.exe"
) else (
    echo [1/2] Using System Python...
    set "PY_CMD=python"
)

echo [2/2] Launching server on http://localhost:8000 ...
"%PY_CMD%" run_app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server terminated unexpectedly.
    pause
)
