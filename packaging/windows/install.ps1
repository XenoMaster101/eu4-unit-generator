# EU4 Unit Generator installer (Track P5): a wizard window, no terminal. Started by "Install EU4 Unit Generator.cmd"
# (or by eu4unitgen-setup.exe, which passes -Quiet -Root).
# Everything goes under ONE folder (default D:\EU4 Unit Generator):
#   app\       the program and its own Python: python.org's signed NuGet build of Python 3.12 (no Python needed on
#              the PC; Smart App Control lets signed Python run), with the app, PySide6, Pillow and Jinja2
#   tools\     portable Blender 4.2 (the app's setup wizard puts it there)
#   ai\        the AI environment, models and weights (setup wizard)
#   wsl\       the Ubuntu distribution TRELLIS.2 runs in (setup wizard)
#   work\      references, projects, result sheets, exported mods
#   settings\  settings, logs, temp files
# Outside it: only the Start-menu and Desktop shortcuts and the entry in Windows' Apps list (with the uninstaller).
# -Quiet installs with no window (tests, setup.exe); -NoSystem skips the shortcuts and the Apps entry (tests on a used PC).
param(
    [string]$Root = "",
    [switch]$Quiet,
    [switch]$NoLaunch,
    [switch]$NoDesktop,
    [switch]$NoStartMenu,
    [switch]$NoSystem
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$AppName = "EU4 Unit Generator"
$Folders = "app", "tools", "ai", "wsl", "work", "settings"
$Version = (Get-ChildItem $Here -Filter "eu4unitgen-*.whl" | Select-Object -First 1).Name -replace '^eu4unitgen-([^-]+)-.*$', '$1'

function Default-Root {
    if (Test-Path "D:\") { return "D:\$AppName" }
    $best = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -gt 0 } | Sort-Object Free -Descending | Select-Object -First 1
    return "$($best.Root)$AppName"
}
if (-not $Root) { $Root = Default-Root }

function Check-Root($root) {
    if (-not (Test-Path $root)) { return $null }
    if (Test-Path (Join-Path $root "app\layout.json")) { return $null }  # ours: an update
    # ours after an uninstall that kept the user's work (its default): only our own folders are left
    $other = @(Get-ChildItem $root -Force -ErrorAction SilentlyContinue | Where-Object { $Folders -notcontains $_.Name })
    if ($other.Count -gt 0) {
        return "$root already has other files in it. Choose an empty folder or a new one."
    }
    return $null
}

function Install-App($root, $desktop, $startMenu, $system, $log) {
    $bad = $(if ($script:LogFile) { $null } else { Check-Root $root })  # quiet mode checked before making its log
    if ($bad) { throw $bad }
    $log.Invoke("Installing $AppName $Version into $root")
    $wheel = Get-ChildItem $Here -Filter "eu4unitgen-*.whl" | Select-Object -First 1
    $nupkg = Get-ChildItem $Here -Filter "python.*.nupkg" | Select-Object -First 1
    if (-not $wheel -or -not $nupkg) { throw "The download is incomplete: unzip the whole folder and run the installer from it." }
    foreach ($f in $Folders) { New-Item -ItemType Directory -Force (Join-Path $root $f) | Out-Null }
    $app = Join-Path $root "app"
    $tmp = Join-Path $root "settings\tmp"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $env:PIP_CACHE_DIR = Join-Path $tmp "pip"   # nothing in the user profile
    $env:PYTHONNOUSERSITE = "1"
    $env:TEMP = $tmp; $env:TMP = $tmp
    $py = Join-Path $app "python\python.exe"
    if (-not (Test-Path $py)) {
        $log.Invoke("Unpacking the app's own Python ($($nupkg.Name))")
        $x = Join-Path $tmp ("python_" + [guid]::NewGuid())
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($nupkg.FullName, $x)
        Move-Item (Join-Path $x "tools") (Join-Path $app "python")
        Remove-Item -Recurse -Force $x
        $sig = Get-AuthenticodeSignature $py
        if ($sig.Status -ne "Valid") { throw "The bundled Python isn't validly signed ($($sig.Status)); the download may be damaged." }
        $log.Invoke("Python is signed by: " + $sig.SignerCertificate.Subject.Split(",")[0])
        Run $py @("-m", "ensurepip", "--default-pip") $log "Setting up pip"
    }
    Run $py @("-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--upgrade",
              "--force-reinstall", "--no-deps", $wheel.FullName) $log "Installing the app"
    Run $py @("-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", $wheel.FullName) $log `
        "Installing its libraries (PySide6, Pillow, Jinja2; about 260 MB to download)"
    Remove-Item -Recurse -Force (Join-Path $tmp "pip") -ErrorAction SilentlyContinue
    foreach ($f in "eu4unitgen.ico", "uninstall.ps1", "LICENSE.txt", "..\LICENSE.txt") {
        if (Test-Path (Join-Path $Here $f)) { Copy-Item (Join-Path $Here $f) $app -Force }
    }
    # the marker: this folder is the app's root (core/layout.py); what the app later makes outside is recorded here
    $markerPath = Join-Path $app "layout.json"
    $m = if (Test-Path $markerPath) { Get-Content $markerPath -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
    $m | Add-Member -NotePropertyName version -NotePropertyValue $Version -Force
    $m | Add-Member -NotePropertyName installed -NotePropertyValue (Get-Date -Format s) -Force
    [IO.File]::WriteAllText($markerPath, ($m | ConvertTo-Json), (New-Object System.Text.UTF8Encoding($false)))
    Run $py @("-c", "from eu4unitgen.core import layout, config; import eu4unitgen, PySide6; print('eu4unitgen', eu4unitgen.__version__, 'root', layout.root(), 'settings', config.config_dir())") $log "Checking it starts"
    $pyw = Join-Path $app "python\pythonw.exe"
    if ($system) {
        $icon = Join-Path $app "eu4unitgen.ico"
        $shell = New-Object -ComObject WScript.Shell
        $places = @()
        if ($startMenu) { $places += [Environment]::GetFolderPath("Programs") }
        if ($desktop) { $places += [Environment]::GetFolderPath("Desktop") }
        foreach ($d in $places) {
            $lnk = $shell.CreateShortcut((Join-Path $d "$AppName.lnk"))
            $lnk.TargetPath = $pyw
            $lnk.Arguments = "-m eu4unitgen.gui"
            $lnk.WorkingDirectory = $root
            $lnk.IconLocation = "$icon,0"
            $lnk.Description = "Make Europa Universalis IV units from your images"
            $lnk.Save()
            $log.Invoke("Shortcut: " + (Join-Path $d "$AppName.lnk"))
        }
        # Windows' Apps list: this user's (no administrator rights needed)
        $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen"
        New-Item -Path $key -Force | Out-Null
        $size = [int]((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1KB)
        $values = @{ DisplayName = $AppName; DisplayVersion = $Version; Publisher = "eu4unitgen (GPL-3.0)";
                     DisplayIcon = $icon; InstallLocation = $root; EstimatedSize = $size; NoModify = 1; NoRepair = 1;
                     UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$app\uninstall.ps1`"" }
        foreach ($k in $values.Keys) {
            $type = if ($values[$k] -is [int]) { "DWord" } else { "String" }
            New-ItemProperty -Path $key -Name $k -Value $values[$k] -PropertyType $type -Force | Out-Null
        }
        $log.Invoke("Added to Windows' Apps list (with its uninstaller).")
    }
    $log.Invoke("Installed.")
    return $pyw
}

function Run($exe, $argv, $log, $title) {
    $log.Invoke($title + "...")
    $out = Join-Path $env:TEMP "install_step.log"
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
    $err = if (Test-Path "$out.err") { (Get-Content "$out.err" | Select-Object -Last 5) -join " " } else { "" }
    Remove-Item $out, "$out.err" -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0) { throw "$title failed (exit $($p.ExitCode)). $err" }
}

if ($Quiet) {
    $bad = Check-Root $Root
    if ($bad) { Write-Host $bad; exit 1 }
    $logDir = Join-Path $Root "settings\logs"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $script:LogFile = Join-Path $logDir "install.log"
    $log = { param($m) Write-Host $m; Add-Content -LiteralPath $script:LogFile -Value $m }
    try {
        $pyw = Install-App $Root (-not $NoDesktop) (-not $NoStartMenu) (-not $NoSystem) $log
    } catch {
        $log.Invoke("The installation stopped: " + $_.Exception.Message)
        exit 1
    }
    if (-not $NoLaunch) { Start-Process $pyw -ArgumentList "-m", "eu4unitgen.gui" -WorkingDirectory $Root }
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
$Form.Font = New-Object System.Drawing.Font("Segoe UI", 10)

$pages = @()
function New-Page { $p = New-Object System.Windows.Forms.Panel; $p.Location = '20,20'; $p.Size = '590,340'; $Form.Controls.Add($p); $p.Visible = $false; $script:pages += $p; return $p }
function Add-Label($page, $text, $y, $h = 24, $bold = $false) {
    $l = New-Object System.Windows.Forms.Label; $l.Text = $text; $l.Location = "0,$y"; $l.Size = "590,$h"
    if ($bold) { $l.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold) }
    $page.Controls.Add($l); return $l
}

# 1. welcome and licence
$p1 = New-Page
if (Test-Path "$Here\icon.png") {  # the PNG: the .ico's sizes are PNG-compressed, which Icon.ToBitmap can't draw
    $pic = New-Object System.Windows.Forms.PictureBox; $pic.Image = [System.Drawing.Image]::FromFile("$Here\icon.png")
    $pic.SizeMode = "Zoom"; $pic.Location = '0,0'; $pic.Size = '64,64'; $p1.Controls.Add($pic)
}
$t = Add-Label $p1 "Welcome to $AppName" 8 32 $true; $t.Location = '80,14'; $t.Size = '500,32'
Add-Label $p1 "Makes Europa Universalis IV units from your own images. This installs the app; the first time it starts, its setup wizard adds Blender, WSL with TRELLIS.2 and the AI weights, all in the same folder." 80 60 | Out-Null
Add-Label $p1 "The app is free software under the GNU General Public License v3:" 150 | Out-Null
$lic = New-Object System.Windows.Forms.TextBox; $lic.Multiline = $true; $lic.ReadOnly = $true; $lic.ScrollBars = "Vertical"
$lic.Location = '0,176'; $lic.Size = '590,160'; $lic.Font = New-Object System.Drawing.Font("Consolas", 8)
$licFile = @("$Here\LICENSE.txt", "$Here\..\LICENSE.txt") | Where-Object { Test-Path $_ } | Select-Object -First 1
$lic.Text = if ($licFile) { (Get-Content $licFile -Raw) -replace "`r?`n", "`r`n" } else { "GPL-3.0" }
$p1.Controls.Add($lic)

# 2. where and shortcuts
$p2 = New-Page
Add-Label $p2 "Where to install" 0 32 $true | Out-Null
Add-Label $p2 "Everything goes in this one folder: the app, Blender, the AI parts and Linux (about 60 GB in all), your work and the settings." 40 44 | Out-Null
$loc = New-Object System.Windows.Forms.TextBox; $loc.Text = $Root; $loc.Location = '0,90'; $loc.Size = '480,26'; $p2.Controls.Add($loc)
$browse = New-Object System.Windows.Forms.Button; $browse.Text = "Browse..."; $browse.Location = '490,88'; $browse.Size = '100,30'; $p2.Controls.Add($browse)
$free = Add-Label $p2 "" 124
function Show-Free { try { $drv = Get-PSDrive ([IO.Path]::GetPathRoot($loc.Text).Substring(0, 1)); $free.Text = "Free on that drive: {0:N0} GB" -f ($drv.Free / 1GB) } catch { $free.Text = "" } }
$loc.Add_TextChanged({ Show-Free }); Show-Free
$browse.Add_Click({ $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.SelectedPath = $loc.Text
    if ($d.ShowDialog() -eq "OK") { $loc.Text = Join-Path $d.SelectedPath $AppName } })
Add-Label $p2 "Inside it: app, tools, ai, wsl, work (your references, projects, result sheets, exported mods) and settings. Nothing else is put on your PC except the shortcuts below and the entry in Windows' Apps list." 156 64 | Out-Null
$cbStart = New-Object System.Windows.Forms.CheckBox; $cbStart.Text = "Start-menu shortcut"; $cbStart.Checked = $true; $cbStart.Location = '0,236'; $cbStart.Size = '300,26'; $p2.Controls.Add($cbStart)
$cbDesk = New-Object System.Windows.Forms.CheckBox; $cbDesk.Text = "Desktop shortcut"; $cbDesk.Checked = $true; $cbDesk.Location = '0,266'; $cbDesk.Size = '300,26'; $p2.Controls.Add($cbDesk)

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
        $bad = Check-Root $loc.Text
        if ($bad) { [System.Windows.Forms.MessageBox]::Show($bad, $AppName, "OK", "Warning") | Out-Null; return }
        Show-Page 2
        $log = { param($m) $box.AppendText($m + "`r`n"); [System.Windows.Forms.Application]::DoEvents() }
        try {
            $script:pyw = Install-App $loc.Text $cbDesk.Checked $cbStart.Checked $true $log
            $doneText.Text = "$AppName is installed in $($loc.Text).`r`n`r`nTo remove it later: Windows Settings > Apps > $AppName > Uninstall. It asks before deleting your work."
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
$Form.Add_Shown({ $lic.SelectionLength = 0; $next.Focus() })
[void]$Form.ShowDialog()
