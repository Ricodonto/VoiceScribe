@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title VoiceScribe - Continuous Voice-to-Notes Router for Obsidian

echo =====================================================================
echo          VOICESCRIBE: CONTINUOUS VOICE-TO-NOTES FOR OBSIDIAN         
echo =====================================================================
echo.

:: 1. Check Python virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Creating virtual environment and installing dependencies...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is in your PATH.
        pause
        exit /b 1
    )
    echo Installing dependencies from requirements.txt...
    .\venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

:: 1b. Create config.json from the example on first run
if not exist "config.json" copy /y "config.example.json" "config.json" >nul

:: 2. Check FFmpeg on PATH
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARNING] FFmpeg was not found in your system PATH!
    echo Audio archival to Opus requires FFmpeg. Please install FFmpeg or add it to PATH.
    echo.
)

:: 3. Check Ollama Connectivity (127.0.0.1:11434)
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $res = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Ollama server is not reachable at http://127.0.0.1:11434
    echo Attempting to launch Ollama in the background...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: 4. Options Menu Selection
if "%1"=="--menu" goto open_menu
if "%1"=="-m" goto open_menu

echo Options:
echo   [1] Start VoiceScribe Recording (Default)
echo   [2] Configure Settings (Microphone, Whisper, Ollama, Pause)
echo.
choice /c 12 /t 4 /d 1 /m "Select option (Auto-starting recording in 4s)"
if errorlevel 2 goto open_menu
goto launch

:open_menu
.\venv\Scripts\python.exe config_menu.py
echo.
echo Launching VoiceScribe with selected configuration...
timeout /t 2 /nobreak >nul
goto launch

:launch
echo Starting VoiceScribe Service...
echo Logs and notes will be generated automatically.
echo Press Ctrl+C in this window at any time to gracefully stop.
echo.

.\venv\Scripts\python.exe scribe_router.py %*

echo.
echo VoiceScribe session closed.
pause

