# EU4 Unit Generator uninstaller (Windows Settings > Apps > Uninstall runs it): removes the app's files, shortcuts and
# its Apps-list entry. It asks before deleting your work (Documents\eu4unitgen), the AI folder (weights and the Windows
# AI environment) and the TRELLIS.2 environment inside WSL: each is kept unless ticked. -Quiet removes only the app.
param([switch]$Quiet)
$Prefix = $PSScriptRoot
$AppName = "EU4 Unit Generator"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

function Size-Of($p) {
    if (-not (Test-Path $p)) { return 0 }
    return [math]::Round(((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB), 1)
}
$settings = Join-Path $env:APPDATA "eu4unitgen\settings.json"
$cfg = if (Test-Path $settings) { Get-Content $settings -Raw | ConvertFrom-Json } else { $null }
$docs = [Environment]::GetFolderPath("MyDocuments")
$work = if ($cfg -and $cfg.workspace) { $cfg.workspace } else { Join-Path $docs "eu4unitgen" }
$ai = if ($cfg -and $cfg.ai_root) { $cfg.ai_root } else { $null }

$delWork = $false; $delAi = $false; $delWsl = $false; $delSettings = $false
if (-not $Quiet) {
    $f = New-Object System.Windows.Forms.Form
    $f.Text = "Uninstall $AppName"; $f.Size = '560,330'; $f.StartPosition = "CenterScreen"; $f.FormBorderStyle = "FixedDialog"
    $f.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    if (Test-Path "$Prefix\eu4unitgen.ico") { $f.Icon = New-Object System.Drawing.Icon("$Prefix\eu4unitgen.ico") }
    $l = New-Object System.Windows.Forms.Label; $l.Text = "Remove $AppName from $Prefix?`r`nYour work and the AI parts are kept unless you tick them:"
    $l.Location = '16,14'; $l.Size = '520,44'; $f.Controls.Add($l)
    $y = 64
    $cbW = New-Object System.Windows.Forms.CheckBox; $cbW.Text = "Also delete my work: $work ($(Size-Of $work) GB)"; $cbW.Location = "16,$y"; $cbW.Size = '520,26'; $cbW.Enabled = (Test-Path $work); $f.Controls.Add($cbW); $y += 32
    $cbA = New-Object System.Windows.Forms.CheckBox; $cbA.Text = if ($ai) { "Also delete the AI folder: $ai ($(Size-Of $ai) GB)" } else { "Also delete the AI folder (none set)" }; $cbA.Location = "16,$y"; $cbA.Size = '520,26'; $cbA.Enabled = [bool]$ai; $f.Controls.Add($cbA); $y += 32
    $cbL = New-Object System.Windows.Forms.CheckBox; $cbL.Text = "Also delete the TRELLIS.2 environment inside WSL (~/eu4unitgen_ai)"; $cbL.Location = "16,$y"; $cbL.Size = '520,26'; $f.Controls.Add($cbL); $y += 32
    $cbS = New-Object System.Windows.Forms.CheckBox; $cbS.Text = "Also delete my settings and logs ($env:APPDATA\eu4unitgen)"; $cbS.Location = "16,$y"; $cbS.Size = '520,26'; $f.Controls.Add($cbS); $y += 44
    $ok = New-Object System.Windows.Forms.Button; $ok.Text = "Uninstall"; $ok.Location = "330,$y"; $ok.Size = '100,32'; $ok.DialogResult = "OK"; $f.Controls.Add($ok)
    $no = New-Object System.Windows.Forms.Button; $no.Text = "Cancel"; $no.Location = "436,$y"; $no.Size = '100,32'; $no.DialogResult = "Cancel"; $f.Controls.Add($no)
    $f.AcceptButton = $ok; $f.CancelButton = $no
    if ($f.ShowDialog() -ne "OK") { exit 0 }
    $delWork = $cbW.Checked; $delAi = $cbA.Checked; $delWsl = $cbL.Checked; $delSettings = $cbS.Checked
}
Set-Location $env:TEMP
foreach ($d in [Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs")) {
    Remove-Item (Join-Path $d "$AppName.lnk") -ErrorAction SilentlyContinue
}
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen" -Recurse -ErrorAction SilentlyContinue
if ($delWork) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
if ($delAi -and $ai) { Remove-Item -Recurse -Force $ai -ErrorAction SilentlyContinue }
if ($delWsl) { wsl.exe -e bash -lc "rm -rf ~/eu4unitgen_ai" 2>$null }
if ($delSettings) { Remove-Item -Recurse -Force (Join-Path $env:APPDATA "eu4unitgen") -ErrorAction SilentlyContinue }
# the app's own folder last (this script lives in it: PowerShell has already read it)
Remove-Item -Recurse -Force $Prefix -ErrorAction SilentlyContinue
$kept = @()
if (-not $delWork -and (Test-Path $work)) { $kept += "your work in $work" }
if (-not $delAi -and $ai) { $kept += "the AI folder $ai" }
$msg = "$AppName is removed." + $(if ($kept.Count) { " Kept: " + ($kept -join "; ") + "." } else { "" })
if (-not $Quiet) { [System.Windows.Forms.MessageBox]::Show($msg, $AppName) | Out-Null } else { Write-Host $msg }
