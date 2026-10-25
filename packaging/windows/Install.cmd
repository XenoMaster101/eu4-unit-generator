@echo off
rem EU4 Unit Generator: double-click to install. Opens the setup wizard window (install.ps1 next to this file).
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0install.ps1" %*
