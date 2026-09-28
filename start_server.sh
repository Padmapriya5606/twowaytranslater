#!/usr/bin/env bash
# Linux / macOS Startup Script for ISL Two-Way Translator
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  Starting ISL Two-Way Translator Web Application"
echo "================================================================="

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    exit 1
fi

echo "[1/2] Checking Python version..."
python3 --version

echo "[2/2] Launching server on http://localhost:8000 ..."
python3 run_app.py
