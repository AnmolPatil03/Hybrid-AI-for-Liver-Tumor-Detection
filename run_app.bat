@echo off
REM Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173)
cd /d "%~dp0"
start "HCC Backend :8000" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "HCC Frontend :5173" cmd /k "cd /d "%~dp0frontend" && npm run dev"
