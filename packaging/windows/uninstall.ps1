# eu4unitgen uninstaller: removes the app (its folder and shortcuts). Your settings (%APPDATA%\eu4unitgen), your runs
# (%LOCALAPPDATA%\eu4unitgen\work), WSL, the AI folder and Blender are kept; the message says where they are.
param([string]$Prefix = "$env:LOCALAPPDATA\eu4unitgen\app", [switch]$Quiet)
Add-Type -AssemblyName PresentationFramework
if (-not $Quiet) {
    $r = [System.Windows.MessageBox]::Show("Remove EU4 Unit Generator from $Prefix? Your settings, runs and the AI setup are kept.",
        "EU4 Unit Generator", "YesNo", "Question")
    if ($r -ne "Yes") { exit 0 }
}
foreach ($d in [Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs")) {
    Remove-Item (Join-Path $d "EU4 Unit Generator.lnk") -ErrorAction SilentlyContinue
}
Set-Location $env:TEMP
Remove-Item -Recurse -Force $Prefix -ErrorAction SilentlyContinue
$msg = "Removed. Kept: settings in $env:APPDATA\eu4unitgen, runs in $env:LOCALAPPDATA\eu4unitgen\work; WSL, the AI folder and Blender as they are."
Write-Host $msg
if (-not $Quiet) { [System.Windows.MessageBox]::Show($msg, "EU4 Unit Generator") | Out-Null }
