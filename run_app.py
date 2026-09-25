"""
Application Runner for OmniRestorer AI Suite
Launches both the FastAPI backend and Vite React frontend concurrently.
"""

import os
import sys
import subprocess
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON_EXE = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


def start_services():
    print("=" * 65)
    print("  OmniRestorer AI Suite - Multi-Task Generative System")
    print("=" * 65)

    # 1. Start FastAPI Backend on port 8000
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        PYTHON_EXE,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
    )

    # Allow backend 1.5 seconds to initialize
    time.sleep(1.5)

    # 2. Start React Vite Frontend on port 5173
    print("[2/2] Starting React Frontend on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=FRONTEND_DIR,
    )

    print("\n" + "=" * 65)
    print("  Application running!")
    print("  --> Open your browser at: http://localhost:5173")
    print("  --> API Documentation at: http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C to stop all services.")
    print("=" * 65 + "\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("All services stopped.")


if __name__ == "__main__":
    start_services()
