@echo off
rem EU4 Unit Generator: removes the app (asks before deleting your work or the AI parts). Also in Windows Settings > Apps.
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0uninstall.ps1" %*
