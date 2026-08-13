@echo off
echo Starting SmartVision Pro...

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%env\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo Error: Python environment not found at %PYTHON_EXE%
    pause
    exit /b 1
)

echo Starting Backend Server on http://localhost:8000 ...
start "SmartVision Server" "%PYTHON_EXE%" -m uvicorn server.main:app --host 0.0.0.0 --port 8000

ping 127.0.0.1 -n 4 >nul

echo Starting SmartVision UI on http://localhost:8501 ...
start "SmartVision Dashboard" "%PYTHON_EXE%" -m streamlit run pipeline/app.py --server.port 8501

echo System started in separate windows.
