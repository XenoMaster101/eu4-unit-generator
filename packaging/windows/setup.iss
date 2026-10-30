; eu4unitgen-setup.exe (Track P5): a classic installer for PCs without Smart App Control (it isn't code-signed, so
; Smart App Control refuses it; the zip's "Install EU4 Unit Generator" works there). Built by build_package.py with
; Inno Setup's ISCC:  ISCC /DAppVersion=<v> /DFilesDir=<stage>\files /DLicense=<stage>\LICENSE.txt /DOutDir=<dist> setup.iss
;
; It is only a wizard (licence, the folder, shortcuts) around the same install.ps1 the zip uses: one install logic,
; one uninstaller (install.ps1 registers <root>\app\uninstall.ps1 in Windows' Apps list), one layout:
; everything under the chosen folder (default D:\EU4 Unit Generator; see core/layout.py).
; Signing it later: SignTool line below (SPEC P5: the signing options).

#define AppName "EU4 Unit Generator"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{B3C6F0E2-5D7A-4C1B-9E44-2F8A6D1C7E90}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=eu4unitgen (GPL-3.0)
DefaultDirName={code:DefaultRoot}
AppendDefaultDirName=no
UsePreviousAppDir=no
DirExistsWarning=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
Uninstallable=no
CreateUninstallRegKey=no
LicenseFile={#License}
SetupIconFile=..\..\src\eu4unitgen\gui\eu4unitgen.ico
OutputDir={#OutDir}
OutputBaseFilename=eu4unitgen-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; SignTool=signtool sign /fd sha256 /tr http://timestamp.acs.microsoft.com /td sha256 $f

[Tasks]
Name: "startmenu"; Description: "Start-menu shortcut"
Name: "desktop"; Description: "Desktop shortcut"

[Files]
Source: "{#FilesDir}\*"; DestDir: "{tmp}\files"; Flags: recursesubdirs deleteafterinstall
Source: "{#License}"; DestDir: "{tmp}"; DestName: "LICENSE.txt"; Flags: deleteafterinstall

[Run]
Filename: "{app}\app\python\pythonw.exe"; Parameters: "-m eu4unitgen.gui"; WorkingDir: "{app}"; \
  Description: "Start {#AppName} now"; Flags: postinstall nowait skipifsilent; \
  Check: Installed
; an update started from the app (/SILENT /RELAUNCH=1): the app starts again when it's done
Filename: "{app}\app\python\pythonw.exe"; Parameters: "-m eu4unitgen.gui"; WorkingDir: "{app}"; \
  Flags: nowait; Check: Relaunch

[Code]
var Ok: Boolean;

const UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\eu4unitgen';

// the folder of the copy already installed (install.ps1 records it in Windows' Apps list), or ''
function InstalledRoot: String;
begin
  Result := '';
  if RegQueryStringValue(HKCU, UninstallKey, 'InstallLocation', Result) then
    if not FileExists(AddBackslash(Result) + 'app\layout.json') then Result := '';
end;

function DefaultRoot(Param: String): String;
begin
  Result := InstalledRoot;
  if Result = '' then
    if DirExists('D:\') then Result := 'D:\{#AppName}' else Result := ExpandConstant('{sd}\{#AppName}');
end;

function IsUpdate: Boolean;
begin
  Result := FileExists(AddBackslash(WizardDirValue) + 'app\layout.json');
end;

function Installed: Boolean;
begin
  Result := Ok;
end;

function Relaunch: Boolean;
begin
  Result := Ok and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

function InitializeSetup: Boolean;
var Waited: Integer;
begin
  // an update started from the app: wait (up to a minute) for the app to close (it holds this mutex while open)
  Waited := 0;
  while CheckForMutexes('eu4unitgen_running') and (Waited < 60) do begin
    Sleep(1000);
    Waited := Waited + 1;
  end;
  Result := not CheckForMutexes('eu4unitgen_running');
  if not Result then
    MsgBox('{#AppName} is still open. Close it, then run setup again.', mbError, MB_OK);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectDir) and (InstalledRoot <> '') then
    WizardForm.SelectDirLabel.Caption := '{#AppName} is already installed here. This updates it to {#AppVersion} ' +
      'in place: your settings, units, library, Blender and AI downloads stay.';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Rc: Integer; Params: String;
begin
  if CurStep = ssPostInstall then begin
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\files\install.ps1') +
              '" -Quiet -NoLaunch -Root "' + ExpandConstant('{app}') + '"';
    if not WizardIsTaskSelected('desktop') then Params := Params + ' -NoDesktop';
    if not WizardIsTaskSelected('startmenu') then Params := Params + ' -NoStartMenu';
    if IsUpdate then
      WizardForm.StatusLabel.Caption := 'Updating the app to {#AppVersion} (your settings, units and downloads stay)...'
    else
      WizardForm.StatusLabel.Caption := 'Installing the app and its libraries (about 260 MB to download)...';
    Ok := Exec('powershell.exe', Params, '', SW_HIDE, ewWaitUntilTerminated, Rc) and (Rc = 0);
    if not Ok then
      MsgBox('The installation stopped (exit ' + IntToStr(Rc) + '). Its log: ' +
             ExpandConstant('{app}\settings\logs\install.log') + #13#10 +
             'Check the internet connection and run setup again.', mbError, MB_OK);
  end;
end;
