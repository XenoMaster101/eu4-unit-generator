@echo off
rem EU4 Unit Generator: double-click to install. Opens the installer's window (files\install.ps1).
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0files\install.ps1" %*
