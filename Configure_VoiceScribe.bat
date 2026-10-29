@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title VoiceScribe Configuration & Options Menu

echo =====================================================================
echo                 VOICESCRIBE: CONFIGURATION MENU                     
echo =====================================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Please run Start_Scribe.bat first to set up the environment.
    pause
    exit /b 1
)

.\venv\Scripts\python.exe config_menu.py

pause
