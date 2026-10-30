@echo off
setlocal
title Install the VoiceScribe Obsidian plugin
rem Usage: Install_Plugin.bat "C:\path\to\Your Vault"   (default: ..\Note Rambling)
set "VAULT=%~1"
if "%VAULT%"=="" set "VAULT=%~dp0..\Note Rambling"
if not exist "%VAULT%\.obsidian" (
  echo [ERROR] "%VAULT%" is not an Obsidian vault ^(no .obsidian folder^).
  echo Usage: Install_Plugin.bat "C:\path\to\Your Vault"
  pause
  exit /b 1
)
set "DEST=%VAULT%\.obsidian\plugins\voicescribe"
if not exist "%DEST%" mkdir "%DEST%"
copy /y "%~dp0obsidian_plugin\main.js" "%DEST%\" >nul
copy /y "%~dp0obsidian_plugin\manifest.json" "%DEST%\" >nul
copy /y "%~dp0obsidian_plugin\styles.css" "%DEST%\" >nul
echo Installed VoiceScribe into "%DEST%".
echo In Obsidian: Settings - Community plugins - enable VoiceScribe (or reload it).
pause
