@echo off
rem EU4 Unit Generator: double-click to remove the app (settings, runs and the AI setup are kept).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
