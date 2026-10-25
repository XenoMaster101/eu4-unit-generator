# EU4 Unit Generator uninstaller (Windows Settings > Apps > Uninstall runs it; it lives in <root>\app).
# Removes the whole root folder (the app, portable Blender, the AI parts, the WSL disk, the settings), the shortcuts,
# the Apps-list entry and the WSL distribution the app registered; .wslconfig too if the app wrote it (the marker,
# app\layout.json, records both). It asks first about your work (work\): kept unless ticked.
# -Quiet removes everything but work\ with no window; -Quiet -DeleteWork removes work\ too.
param([switch]$Quiet, [switch]$DeleteWork)
$App = $PSScriptRoot
$Root = Split-Path $App -Parent
$AppName = "EU4 Unit Generator"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

if (-not (Test-Path (Join-Path $App "layout.json"))) {
    $msg = "$App isn't an installed $AppName (no layout.json): nothing removed."
    if ($Quiet) { Write-Host $msg } else { [System.Windows.Forms.MessageBox]::Show($msg, $AppName) | Out-Null }
    exit 1
}
$marker = Get-Content (Join-Path $App "layout.json") -Raw | ConvertFrom-Json
$distro = $marker.wsl_distro
$work = Join-Path $Root "work"
function Size-Of($p) {
    if (-not (Test-Path $p)) { return 0 }
    return [math]::Round(((Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB), 1)
}
# the app must be closed: its files are in use while it runs
$running = Get-Process python, pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    $msg = "Close $AppName first (it's still running), then uninstall again."
    if ($Quiet) { Write-Host $msg } else { [System.Windows.Forms.MessageBox]::Show($msg, $AppName) | Out-Null }
    exit 1
}

$delWork = [bool]$DeleteWork
if (-not $Quiet) {
    $total = Size-Of $Root
    $f = New-Object System.Windows.Forms.Form
    $f.Text = "Uninstall $AppName"; $f.Size = '580,300'; $f.StartPosition = "CenterScreen"; $f.FormBorderStyle = "FixedDialog"
    $f.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    if (Test-Path "$App\eu4unitgen.ico") { $f.Icon = New-Object System.Drawing.Icon("$App\eu4unitgen.ico") }
    $text = "Remove $AppName and everything in $Root ($total GB): the app, Blender, the AI parts" +
            $(if ($distro) { ", the Linux distribution $distro" } else { "" }) + " and the settings?`r`n`r`nYour work (work\: references, projects, result sheets, exported mods) is kept unless you tick:"
    $l = New-Object System.Windows.Forms.Label; $l.Text = $text; $l.Location = '16,14'; $l.Size = '540,110'; $f.Controls.Add($l)
    $cbW = New-Object System.Windows.Forms.CheckBox; $cbW.Text = "Also delete my work ($(Size-Of $work) GB)"; $cbW.Location = '16,128'; $cbW.Size = '540,26'
    $cbW.Enabled = (Test-Path $work); $f.Controls.Add($cbW)
    $ok = New-Object System.Windows.Forms.Button; $ok.Text = "Uninstall"; $ok.Location = '350,200'; $ok.Size = '100,32'; $ok.DialogResult = "OK"; $f.Controls.Add($ok)
    $no = New-Object System.Windows.Forms.Button; $no.Text = "Cancel"; $no.Location = '456,200'; $no.Size = '100,32'; $no.DialogResult = "Cancel"; $f.Controls.Add($no)
    $f.AcceptButton = $no; $f.CancelButton = $no
    if ($f.ShowDialog() -ne "OK") { exit 0 }
    $delWork = $cbW.Checked
}
Set-Location $env:TEMP
foreach ($d in [Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs")) {
    Remove-Item (Join-Path $d "$AppName.lnk") -ErrorAction SilentlyContinue
}
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen" -Recurse -ErrorAction SilentlyContinue
$notes = @()
if ($distro) {  # only the distribution the app registered (never the user's own)
    & wsl.exe --terminate $distro 2>$null | Out-Null
    & wsl.exe --unregister $distro 2>$null | Out-Null
    $notes += "the Linux distribution $distro"
}
if ($marker.wslconfig) {
    Remove-Item (Join-Path $env:USERPROFILE ".wslconfig") -ErrorAction SilentlyContinue
    $notes += ".wslconfig"
}
# the root last; this script lives in it (PowerShell has already read it)
foreach ($item in Get-ChildItem $Root -Force) {
    if ($item.Name -eq "work" -and -not $delWork) { continue }
    Remove-Item -LiteralPath $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
$left = @(Get-ChildItem $Root -Force -ErrorAction SilentlyContinue)
if ($left.Count -eq 0) { Remove-Item -LiteralPath $Root -Force -ErrorAction SilentlyContinue }
$msg = "$AppName is removed" + $(if ($notes.Count) { " (with " + ($notes -join " and ") + ")" } else { "" }) + "." +
       $(if (-not $delWork -and (Test-Path $work)) { " Your work is kept in $work." } else { "" })
$stuck = @(Get-ChildItem $Root -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "work" })
if ($stuck.Count) { $msg += " Some files were in use and are left in ${Root}: delete them after a restart." }
if (-not $Quiet) { [System.Windows.Forms.MessageBox]::Show($msg, $AppName) | Out-Null } else { Write-Host $msg }
