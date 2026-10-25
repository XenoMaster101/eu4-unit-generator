# The fresh-user installer test, run inside Windows Sandbox (a clean, throw-away Windows; fresh_user.wsb starts it).
# Installs the package quietly with shortcuts and the Apps-list entry, checks each one, starts the app for 20 s and
# grabs a screenshot, then uninstalls quietly and checks everything the uninstaller should remove is gone.
# Writes results.txt and app.png to C:\results (mapped to the host's dist\sandbox_results).
# The installer's own wizard window is left on screen afterwards for a person to click through.
$ErrorActionPreference = "Continue"
$R = "C:\results"
New-Item -ItemType Directory -Force $R | Out-Null
$log = Join-Path $R "results.txt"
Set-Content $log "eu4unitgen Sandbox test $(Get-Date -Format s)"
function Check($what, $ok) { Add-Content $log ("{0}  {1}" -f $(if ($ok) { "PASS" } else { "FAIL" }), $what) }

$pkg = Get-ChildItem "C:\pkg" -Filter "eu4unitgen-*-windows.zip" | Select-Object -First 1
Expand-Archive $pkg.FullName -DestinationPath "C:\pkg_unzipped" -Force
$dir = Get-ChildItem "C:\pkg_unzipped" -Directory | Select-Object -First 1
Check "no Python on this PC before install" (-not (Get-Command python.exe -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*WindowsApps*" }))

$prefix = "$env:LOCALAPPDATA\Programs\eu4unitgen"
$t = Get-Date
& powershell -NoProfile -ExecutionPolicy Bypass -File "$($dir.FullName)\install.ps1" -Quiet -NoLaunch *>> (Join-Path $R "install_output.txt")
Add-Content $log ("install took {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)
Check "app folder" (Test-Path "$prefix\python\pythonw.exe")
Check "Python signed by the PSF" ((Get-AuthenticodeSignature "$prefix\python\python.exe").Status -eq "Valid")
Check "Start-menu shortcut" (Test-Path (Join-Path ([Environment]::GetFolderPath("Programs")) "EU4 Unit Generator.lnk"))
Check "Desktop shortcut" (Test-Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "EU4 Unit Generator.lnk"))
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen"
Check "Apps-list entry" (Test-Path $key)
if (Test-Path $key) { Add-Content $log ((Get-ItemProperty $key | Select-Object DisplayName, DisplayVersion, DisplayIcon, EstimatedSize | Out-String).Trim()) }

# start the app from its shortcut target, as a user would
$p = Start-Process "$prefix\python\pythonw.exe" -ArgumentList "-m", "eu4unitgen.gui" -WorkingDirectory $prefix -PassThru
Start-Sleep 20
Check "app still running after 20 s" (-not $p.HasExited)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
[System.Drawing.Graphics]::FromImage($bmp).CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save((Join-Path $R "app.png"))
Check "settings folder made" (Test-Path "$env:APPDATA\eu4unitgen")
Check "Documents\eu4unitgen made" (Test-Path (Join-Path ([Environment]::GetFolderPath("MyDocuments")) "eu4unitgen"))
if (Test-Path "$env:APPDATA\eu4unitgen\logs\gui.log") { Copy-Item "$env:APPDATA\eu4unitgen\logs\gui.log" $R }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 2

& powershell -NoProfile -ExecutionPolicy Bypass -File "$prefix\uninstall.ps1" -Quiet *>> (Join-Path $R "uninstall_output.txt")
Start-Sleep 2
Check "uninstall: app folder gone" (-not (Test-Path $prefix))
Check "uninstall: shortcuts gone" (-not (Test-Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "EU4 Unit Generator.lnk")))
Check "uninstall: Apps entry gone" (-not (Test-Path $key))
Check "uninstall: your work kept (quiet mode keeps it)" (Test-Path (Join-Path ([Environment]::GetFolderPath("MyDocuments")) "eu4unitgen"))
Add-Content $log "done"
# now the wizard window, for a person to click through
Start-Process "$($dir.FullName)\Install.cmd"
