# PowerShell Startup Script for SmartVision Pro
$PythonExe = "env\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "Error: Virtual environment not found at $PythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "Starting Backend Server on http://localhost:8000 ..." -ForegroundColor Green
Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn server.main:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 3

Write-Host "Starting SmartVision Dashboard on http://localhost:8501 ..." -ForegroundColor Green
Start-Process -FilePath $PythonExe -ArgumentList "-m streamlit run pipeline/app.py --server.port 8501"

Write-Host "SmartVision Pro services started." -ForegroundColor Green
