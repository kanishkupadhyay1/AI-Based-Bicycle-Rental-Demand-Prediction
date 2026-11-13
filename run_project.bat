@echo off
setlocal enabledelayedexpansion

echo ============================================================================
echo   AI-Based Bicycle Rental Demand Prediction (GAM) - Project Launcher
echo ============================================================================

REM Check Python availability
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH. Please install Python 3.11+.
    exit /b 1
)

REM Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

echo [INFO] Installing/verifying Python dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Python requirements.
    exit /b 1
)

REM Support optional --force argument to force retraining
set PIPELINE_ARGS=
if /I "%~1"=="--force" (
    echo [INFO] Force retraining enabled.
    set PIPELINE_ARGS=--force
)

echo [INFO] Running ML pipeline (download, validate, preprocess, train, evaluate, plot)...
".venv\Scripts\python.exe" -m src.run_pipeline %PIPELINE_ARGS%
if errorlevel 1 (
    echo [ERROR] ML pipeline execution failed.
    exit /b 1
)

REM Install frontend dependencies if node_modules missing
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend npm dependencies...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] Failed to install frontend dependencies.
        exit /b 1
    )
    popd
)

echo [INFO] Starting FastAPI backend on http://localhost:8000 ...
start "Bike Demand API (FastAPI)" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] Starting React frontend on http://localhost:5173 ...
start "Bike Demand Dashboard (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo ============================================================================
echo   Backend API : http://localhost:8000
echo   API Docs    : http://localhost:8000/docs
echo   Frontend UI : http://localhost:5173
echo   Tip: Run "run_project.bat --force" to force model retraining.
echo ============================================================================
endlocal
