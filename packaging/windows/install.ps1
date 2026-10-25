# EU4 Unit Generator installer (Track P4): a wizard window, no terminal. Started by "Install EU4 Unit Generator.cmd".
# What it does:
#   1. the app's own Python: python.org's signed NuGet build (python-3.12.10, in this folder), unpacked into
#      <location>\python (no Python needed on the PC; Smart App Control lets signed Python run);
#   2. the app (the wheel in this folder) with PySide6, Jinja2 and Pillow from PyPI (about 260 MB);
#   3. Start-menu and Desktop shortcuts with the app's icon; an entry in Windows' Apps list with its uninstaller.
# The app's files go in -Prefix (default %LOCALAPPDATA%\Programs\eu4unitgen: no administrator rights needed). Your
# work goes in Documents\eu4unitgen, settings and logs in %APPDATA%\eu4unitgen; the AI parts where you choose in the
# app's setup wizard. -Quiet installs with the defaults and no window (tests).
param(
    [string]$Prefix = "$env:LOCALAPPDATA\Programs\eu4unitgen",
    [switch]$Quiet,
    [switch]$NoLaunch,
    [switch]$NoSystem  # tests on a used PC: no shortcuts, no Apps-list entry
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$AppName = "EU4 Unit Generator"
$Version = (Get-ChildItem $Here -Filter "eu4unitgen-*.whl" | Select-Object -First 1).Name -replace '^eu4unitgen-([^-]+)-.*$', '$1'

function Install-App($prefix, $desktop, $startMenu, $log) {
    $log.Invoke("Installing $AppName $Version into $prefix")
    $wheel = Get-ChildItem $Here -Filter "eu4unitgen-*.whl" | Select-Object -First 1
    $nupkg = Get-ChildItem $Here -Filter "python.*.nupkg" | Select-Object -First 1
    if (-not $wheel -or -not $nupkg) { throw "The download is incomplete: unzip the whole folder and run the installer from it." }
    New-Item -ItemType Directory -Force $prefix | Out-Null
    $py = Join-Path $prefix "python\python.exe"
    if (-not (Test-Path $py)) {
        $log.Invoke("Unpacking the app's own Python ($($nupkg.Name))")
        $tmp = Join-Path $env:TEMP ("eu4unitgen_py_" + [guid]::NewGuid())
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($nupkg.FullName, $tmp)
        Move-Item (Join-Path $tmp "tools") (Join-Path $prefix "python")
        Remove-Item -Recurse -Force $tmp
        $sig = Get-AuthenticodeSignature $py
        if ($sig.Status -ne "Valid") { throw "The bundled Python isn't validly signed ($($sig.Status)); the download may be damaged." }
        $log.Invoke("Python is signed by: " + $sig.SignerCertificate.Subject.Split(",")[0])
        Run $py @("-m", "ensurepip", "--default-pip") $log "Setting up pip"
    }
    Run $py @("-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--upgrade",
              "--force-reinstall", "--no-deps", $wheel.FullName) $log "Installing the app"
    Run $py @("-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", $wheel.FullName) $log `
        "Installing its libraries (PySide6, Pillow, Jinja2; about 260 MB to download)"
    Run $py @("-c", "import eu4unitgen, PySide6; print('eu4unitgen', eu4unitgen.__version__)") $log "Checking it starts"
    foreach ($f in "eu4unitgen.ico", "uninstall.ps1", "Uninstall.cmd", "LICENSE.txt") {
        if (Test-Path (Join-Path $Here $f)) { Copy-Item (Join-Path $Here $f) $prefix -Force }
    }
    $pyw = Join-Path $prefix "python\pythonw.exe"
    $icon = Join-Path $prefix "eu4unitgen.ico"
    $shell = New-Object -ComObject WScript.Shell
    $places = @()
    if ($startMenu) { $places += [Environment]::GetFolderPath("Programs") }
    if ($desktop) { $places += [Environment]::GetFolderPath("Desktop") }
    foreach ($d in $places) {
        $lnk = $shell.CreateShortcut((Join-Path $d "$AppName.lnk"))
        $lnk.TargetPath = $pyw
        $lnk.Arguments = "-m eu4unitgen.gui"
        $lnk.WorkingDirectory = $prefix
        $lnk.IconLocation = "$icon,0"
        $lnk.Description = "Make Europa Universalis IV units from your images"
        $lnk.Save()
        $log.Invoke("Shortcut: " + (Join-Path $d "$AppName.lnk"))
    }
    # Windows' Apps list: this user's (no administrator rights needed)
    if ($NoSystem) { $log.Invoke("Installed (no shortcuts or Apps-list entry: -NoSystem)."); return $pyw }
    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen"
    New-Item -Path $key -Force | Out-Null
    $size = [int]((Get-ChildItem $prefix -Recurse -File | Measure-Object Length -Sum).Sum / 1KB)
    $values = @{ DisplayName = $AppName; DisplayVersion = $Version; Publisher = "eu4unitgen (GPL-3.0)";
                 DisplayIcon = $icon; InstallLocation = $prefix; EstimatedSize = $size; NoModify = 1; NoRepair = 1;
                 UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$prefix\uninstall.ps1`"" }
    foreach ($k in $values.Keys) {
        $type = if ($values[$k] -is [int]) { "DWord" } else { "String" }
        New-ItemProperty -Path $key -Name $k -Value $values[$k] -PropertyType $type -Force | Out-Null
    }
    $log.Invoke("Added to Windows' Apps list (with its uninstaller).")
    $log.Invoke("Installed.")
    return $pyw
}

function Run($exe, $argv, $log, $title) {
    $log.Invoke($title + "...")
    $out = Join-Path $env:TEMP "eu4unitgen_install_step.log"
    $p = Start-Process -FilePath $exe -ArgumentList ($argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) `
        -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    $null = $p.Handle  # without this, PowerShell loses the exit code
    $seen = 0
    while (-not $p.HasExited) {
        Start-Sleep -Milliseconds 300
        if (Test-Path $out) {
            $lines = @(Get-Content $out -ErrorAction SilentlyContinue)
            for ($i = $seen; $i -lt $lines.Count; $i++) { $log.Invoke("   " + $lines[$i]) }
            $seen = $lines.Count
        }
        if ($script:Form) { [System.Windows.Forms.Application]::DoEvents() }
    }
    $p.WaitForExit()
    if (Test-Path $out) { $lines = @(Get-Content $out); for ($i = $seen; $i -lt $lines.Count; $i++) { $log.Invoke("   " + $lines[$i]) } }
    if ($p.ExitCode -ne 0) {
        $err = if (Test-Path "$out.err") { (Get-Content "$out.err" | Select-Object -Last 5) -join " " } else { "" }
        throw "$title failed (exit $($p.ExitCode)). $err"
    }
}

if ($Quiet) {
    $pyw = Install-App $Prefix (-not $NoSystem) (-not $NoSystem) { param($m) Write-Host $m }
    if (-not $NoLaunch) { Start-Process $pyw -ArgumentList "-m", "eu4unitgen.gui" -WorkingDirectory $Prefix }
    exit 0
}

# ------------------------------------------------------------------ the wizard window
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$script:Form = New-Object System.Windows.Forms.Form
$Form.Text = "$AppName $Version - Setup"
$Form.Size = New-Object System.Drawing.Size(640, 470)
$Form.StartPosition = "CenterScreen"
$Form.FormBorderStyle = "FixedDialog"
$Form.MaximizeBox = $false
if (Test-Path "$Here\eu4unitgen.ico") { $Form.Icon = New-Object System.Drawing.Icon("$Here\eu4unitgen.ico") }
$font = New-Object System.Drawing.Font("Segoe UI", 10)
$Form.Font = $font

$pages = @()
function New-Page { $p = New-Object System.Windows.Forms.Panel; $p.Location = '20,20'; $p.Size = '590,340'; $Form.Controls.Add($p); $p.Visible = $false; $script:pages += $p; return $p }
function Add-Label($page, $text, $y, $h = 24, $bold = $false) {
    $l = New-Object System.Windows.Forms.Label; $l.Text = $text; $l.Location = "0,$y"; $l.Size = "590,$h"
    if ($bold) { $l.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold) }
    $page.Controls.Add($l); return $l
}

# 1. welcome and licence
$p1 = New-Page
if (Test-Path "$Here\eu4unitgen.ico") {
    $pic = New-Object System.Windows.Forms.PictureBox; $pic.Image = (New-Object System.Drawing.Icon("$Here\eu4unitgen.ico", 64, 64)).ToBitmap()
    $pic.Location = '0,0'; $pic.Size = '64,64'; $p1.Controls.Add($pic)
}
$t = Add-Label $p1 "Welcome to $AppName" 8 32 $true; $t.Location = '80,14'; $t.Size = '500,32'
Add-Label $p1 "Makes Europa Universalis IV units from your own images. This installs the app; the first time it starts, its setup wizard walks you through the AI side (WSL, TRELLIS.2, the weights) and Blender." 80 60 | Out-Null
Add-Label $p1 "The app is free software under the GNU General Public License v3:" 150 | Out-Null
$lic = New-Object System.Windows.Forms.TextBox; $lic.Multiline = $true; $lic.ReadOnly = $true; $lic.ScrollBars = "Vertical"
$lic.Location = '0,176'; $lic.Size = '590,160'; $lic.Font = New-Object System.Drawing.Font("Consolas", 8)
$lic.Text = if (Test-Path "$Here\LICENSE.txt") { (Get-Content "$Here\LICENSE.txt" -Raw) -replace "`r?`n", "`r`n" } else { "GPL-3.0" }
$p1.Controls.Add($lic)

# 2. where and shortcuts
$p2 = New-Page
Add-Label $p2 "Where to install" 0 32 $true | Out-Null
Add-Label $p2 "The app's own files (about 700 MB with its Python and libraries):" 44 | Out-Null
$loc = New-Object System.Windows.Forms.TextBox; $loc.Text = $Prefix; $loc.Location = '0,72'; $loc.Size = '480,26'; $p2.Controls.Add($loc)
$browse = New-Object System.Windows.Forms.Button; $browse.Text = "Browse..."; $browse.Location = '490,70'; $browse.Size = '100,30'; $p2.Controls.Add($browse)
$browse.Add_Click({ $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.SelectedPath = $loc.Text
    if ($d.ShowDialog() -eq "OK") { $loc.Text = Join-Path $d.SelectedPath "eu4unitgen" } })
$docs = [Environment]::GetFolderPath("MyDocuments")
Add-Label $p2 "Your work goes in $docs\eu4unitgen (references, projects, result sheets, exported mods); settings and logs in $env:APPDATA\eu4unitgen. You choose where the AI parts go (about 60 GB) in the app's setup wizard." 116 70 | Out-Null
$cbStart = New-Object System.Windows.Forms.CheckBox; $cbStart.Text = "Start-menu shortcut"; $cbStart.Checked = $true; $cbStart.Location = '0,200'; $cbStart.Size = '300,26'; $p2.Controls.Add($cbStart)
$cbDesk = New-Object System.Windows.Forms.CheckBox; $cbDesk.Text = "Desktop shortcut"; $cbDesk.Checked = $true; $cbDesk.Location = '0,230'; $cbDesk.Size = '300,26'; $p2.Controls.Add($cbDesk)

# 3. progress
$p3 = New-Page
Add-Label $p3 "Installing" 0 32 $true | Out-Null
$bar = New-Object System.Windows.Forms.ProgressBar; $bar.Style = "Marquee"; $bar.Location = '0,44'; $bar.Size = '590,20'; $p3.Controls.Add($bar)
$box = New-Object System.Windows.Forms.TextBox; $box.Multiline = $true; $box.ReadOnly = $true; $box.ScrollBars = "Vertical"
$box.Location = '0,74'; $box.Size = '590,262'; $box.Font = New-Object System.Drawing.Font("Consolas", 8); $p3.Controls.Add($box)

# 4. finish
$p4 = New-Page
Add-Label $p4 "Done" 0 32 $true | Out-Null
$doneText = Add-Label $p4 "" 44 120
$cbLaunch = New-Object System.Windows.Forms.CheckBox; $cbLaunch.Text = "Start $AppName now (its setup wizard opens)"; $cbLaunch.Checked = $true
$cbLaunch.Location = '0,180'; $cbLaunch.Size = '500,26'; $p4.Controls.Add($cbLaunch)

$back = New-Object System.Windows.Forms.Button; $back.Text = "< Back"; $back.Location = '330,380'; $back.Size = '90,32'
$next = New-Object System.Windows.Forms.Button; $next.Text = "Next >"; $next.Location = '425,380'; $next.Size = '90,32'
$cancel = New-Object System.Windows.Forms.Button; $cancel.Text = "Cancel"; $cancel.Location = '520,380'; $cancel.Size = '90,32'
$Form.Controls.AddRange(@($back, $next, $cancel))
$script:page = 0
$script:pyw = $null
function Show-Page($i) {
    for ($k = 0; $k -lt $pages.Count; $k++) { $pages[$k].Visible = ($k -eq $i) }
    $script:page = $i
    $back.Enabled = ($i -eq 1)
    $next.Text = @("Next >", "Install", "Next >", "Finish")[$i]
    $next.Enabled = ($i -ne 2)
    $cancel.Enabled = ($i -lt 2)
}
$cancel.Add_Click({ $Form.Close() })
$back.Add_Click({ Show-Page ($script:page - 1) })
$next.Add_Click({
    if ($script:page -eq 0) { Show-Page 1; return }
    if ($script:page -eq 1) {
        Show-Page 2
        $log = { param($m) $box.AppendText($m + "`r`n"); [System.Windows.Forms.Application]::DoEvents() }
        try {
            $script:pyw = Install-App $loc.Text $cbDesk.Checked $cbStart.Checked $log
            $doneText.Text = "$AppName is installed in $($loc.Text).`r`n`r`nTo remove it later: Windows Settings > Apps > $AppName > Uninstall. Your work and the AI parts are only deleted if you say so then."
            $bar.Style = "Continuous"; $bar.Value = 100
            Show-Page 3
        } catch {
            $bar.Style = "Continuous"; $bar.Value = 0
            $log.Invoke("`r`nThe installation stopped: " + $_.Exception.Message)
            [System.Windows.Forms.MessageBox]::Show("The installation stopped:`r`n" + $_.Exception.Message + "`r`n`r`nCheck the internet connection and run the installer again.", $AppName, "OK", "Error") | Out-Null
            $cancel.Enabled = $true; $cancel.Text = "Close"
        }
        return
    }
    if ($script:page -eq 3) {
        if ($cbLaunch.Checked -and $script:pyw) { Start-Process $script:pyw -ArgumentList "-m", "eu4unitgen.gui" -WorkingDirectory $loc.Text }
        $Form.Close()
    }
})
Show-Page 0
[void]$Form.ShowDialog()
