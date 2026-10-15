<#
.SYNOPSIS
  Create (or recreate) the desktop shortcut that starts the eu4unitgen GUI with no console window.

.DESCRIPTION
  The shortcut runs  <repo>\.venv\Scripts\pythonw.exe -m eu4unitgen.gui  with the repo as its
  working folder (relative defaults such as output\ and build\test_mod resolve there).
  pythonw.exe is the windowless Python, so no console opens.
  Windows only. Run from anywhere:
      powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
  Options: -Name "EU4 Unit Generator"  -Folder <where to put it; default: your Desktop>
#>
param(
    [string]$Name = "EU4 Unit Generator",
    [string]$Folder = [Environment]::GetFolderPath("Desktop")
)
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    throw "No $pythonw. Create the venv first: python -m venv .venv; .venv\Scripts\pip install -e .[dev]"
}

$link = Join-Path $Folder "$Name.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)   # overwrites an existing shortcut of the same name
$sc.TargetPath = $pythonw
$sc.Arguments = "-m eu4unitgen.gui"
$sc.WorkingDirectory = $repo
$sc.Description = "eu4unitgen: EU4 unit set generator (Hellenistic Revolution)"
$sc.IconLocation = "$pythonw,0"
$sc.WindowStyle = 1
$sc.Save()
Write-Output "Shortcut: $link"
Write-Output "  runs: `"$pythonw`" -m eu4unitgen.gui   (in $repo)"
