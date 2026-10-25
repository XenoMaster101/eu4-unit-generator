# eu4unitgen installer for Windows (Track P4). Started by Install.cmd; nothing is typed: questions are message boxes.
#  1. finds Python 3.11 or newer (the py launcher, then python on PATH); if there is none, offers to install
#     Python 3.12 for this user with winget;
#  2. makes a virtual environment in -Prefix (default %LOCALAPPDATA%\eu4unitgen\app) and installs the app's wheel
#     (with PySide6, Jinja2 and Pillow from PyPI);
#  3. puts "EU4 Unit Generator" on the Desktop and in the Start menu (pythonw: no console window);
#  4. starts the app (its Setup screen opens first on a new machine: WSL, TRELLIS.2, the AI weights, Blender).
# Settings (%APPDATA%\eu4unitgen) and runs (%LOCALAPPDATA%\eu4unitgen\work) are never touched by reinstalling.
param(
    [string]$Prefix = "$env:LOCALAPPDATA\eu4unitgen\app",
    [switch]$NoShortcuts,
    [switch]$NoLaunch,
    [switch]$Quiet
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName PresentationFramework

function Ask($text) {
    if ($Quiet) { return $true }
    return [System.Windows.MessageBox]::Show($text, "EU4 Unit Generator", "YesNo", "Question") -eq "Yes"
}
function Say($text, $icon = "Information") {
    Write-Host $text
    if (-not $Quiet) { [System.Windows.MessageBox]::Show($text, "EU4 Unit Generator", "OK", $icon) | Out-Null }
}
function Good-Python($exe, $args0) {
    try {
        $out = & $exe @args0 -c "import sys; print(int(sys.version_info >= (3, 11)))" 2>$null
        return ($out -eq "1")
    } catch { return $false }
}
function Find-Python {
    foreach ($v in "3.13", "3.12", "3.11", "3.14") {
        if ((Get-Command py -ErrorAction SilentlyContinue) -and (Good-Python "py" @("-$v"))) { return @("py", "-$v") }
    }
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p -and $p.Source -notlike "*WindowsApps*" -and (Good-Python $p.Source @())) { return @($p.Source) }
    return $null
}

try {
    Write-Host "EU4 Unit Generator: installing into $Prefix"
    $wheel = Get-ChildItem -Path $PSScriptRoot -Filter "eu4unitgen-*.whl" | Sort-Object Name | Select-Object -Last 1
    if (-not $wheel) { throw "The app's package (eu4unitgen-*.whl) isn't next to this script. Unzip the whole download first." }

    $py = Find-Python
    if (-not $py) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "Python 3.11 or newer wasn't found. Install it from python.org (tick 'Add python.exe to PATH'), then run Install.cmd again."
        }
        if (-not (Ask "Python 3.11 or newer wasn't found. Install Python 3.12 for your user now (winget, about 2 minutes)?")) {
            throw "Python is needed. Install Python 3.11 or newer from python.org, then run Install.cmd again."
        }
        winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
        $py = Find-Python
        if (-not $py) { throw "Python was installed but can't be found yet. Sign out and in again, then run Install.cmd again." }
    }
    Write-Host "Python: $($py -join ' ')"

    $venv = Join-Path $Prefix "venv"
    New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Write-Host "Making the app's Python environment..."
        $exe = $py[0]; $rest = @(); if ($py.Count -gt 1) { $rest = $py[1..($py.Count - 1)] }
        & $exe @rest -m venv $venv
    }
    $vpy = Join-Path $venv "Scripts\python.exe"
    Write-Host "Installing the app and its libraries (PySide6, Jinja2, Pillow)..."
    & $vpy -m pip install --disable-pip-version-check --upgrade pip | Out-Null
    & $vpy -m pip install --disable-pip-version-check --upgrade --force-reinstall --no-deps $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "Installing the app failed (pip exit $LASTEXITCODE)." }
    & $vpy -m pip install --disable-pip-version-check $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "Installing the app's libraries failed (pip exit $LASTEXITCODE). Check the internet connection." }
    & $vpy -c "import eu4unitgen, PySide6, PIL, jinja2; print('eu4unitgen', eu4unitgen.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "The app doesn't start in its environment." }
    Copy-Item (Join-Path $PSScriptRoot "Uninstall.cmd") $Prefix -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $PSScriptRoot "uninstall.ps1") $Prefix -Force -ErrorAction SilentlyContinue

    $pyw = Join-Path $venv "Scripts\pythonw.exe"
    if (-not $NoShortcuts) {
        $shell = New-Object -ComObject WScript.Shell
        $places = @([Environment]::GetFolderPath("Desktop"), (Join-Path ([Environment]::GetFolderPath("Programs")) ""))
        foreach ($d in $places) {
            $lnk = $shell.CreateShortcut((Join-Path $d "EU4 Unit Generator.lnk"))
            $lnk.TargetPath = $pyw
            $lnk.Arguments = "-m eu4unitgen.gui"
            $lnk.WorkingDirectory = $Prefix
            $lnk.Description = "Make Europa Universalis IV units from your images"
            $lnk.Save()
        }
    }
    Write-Host "Installed."
    if (-not $NoLaunch) {
        Start-Process -FilePath $pyw -ArgumentList "-m", "eu4unitgen.gui" -WorkingDirectory $Prefix
        Say "EU4 Unit Generator is installed and starting. Its Setup screen walks you through the AI side (WSL, TRELLIS.2, the weights). Shortcuts are on the Desktop and in the Start menu."
    }
    exit 0
} catch {
    Say ("The installation stopped: " + $_.Exception.Message) "Error"
    exit 1
}
