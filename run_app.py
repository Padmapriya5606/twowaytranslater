#!/usr/bin/env python3
"""
ISL Two-Way Translator — Unified Server Launcher
Starts the FastAPI backend & serves the interactive modern Web App.
"""
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    import uvicorn
    from backend.app import app
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("RAILWAY_PORT", 8000)))
    
    print("=" * 65)
    print(">> ISL Two-Way Translator (Live Web Application & API)")
    print(f">> Host: {host} | Port: {port}")
    print(f">> API Docs: http://{host}:{port}/docs")
    print("=" * 65)
    
    uvicorn.run(app, host=host, port=port, reload=False)
