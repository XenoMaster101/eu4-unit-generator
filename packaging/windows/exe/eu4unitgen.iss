; Inno Setup script for a signed eu4unitgen-setup.exe (Track P4). NOT used today: Smart App Control refuses unsigned
; .exe files, so the shipped package is install.ps1 (packaging/windows/build_package.py). Use this with a code-signing
; certificate (SPEC §P4: the signing options).
;
; Build: first the frozen app (pyinstaller packaging/windows/exe/eu4unitgen.spec), sign dist\eu4unitgen\eu4unitgen.exe,
; then:  iscc packaging\windows\exe\eu4unitgen.iss   ->  dist\eu4unitgen-setup.exe  (sign it too; SignTool below)
;
; The same layout as install.ps1: the app in %LOCALAPPDATA%\Programs\eu4unitgen (no administrator rights), your work in
; Documents\eu4unitgen, settings and logs in %APPDATA%\eu4unitgen, the AI parts where the app's setup wizard puts them.
; The uninstaller removes the app; it asks before deleting your work. (The AI folder is kept: install.ps1's uninstaller offers it.)

#define AppName "EU4 Unit Generator"
#define AppVersion "0.1.0"
#define Root "..\..\.."

[Setup]
AppId={{264D0193-4A99-4595-9024-A60F7C0A6ED0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=eu4unitgen (GPL-3.0)
DefaultDirName={localappdata}\Programs\eu4unitgen
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile={#Root}\LICENSE
SetupIconFile={#Root}\src\eu4unitgen\gui\eu4unitgen.ico
UninstallDisplayIcon={app}\eu4unitgen.exe
OutputDir={#Root}\dist
OutputBaseFilename=eu4unitgen-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
; SignTool=signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /a $f

[Tasks]
Name: "desktopicon"; Description: "Desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Root}\dist\eu4unitgen\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Root}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\eu4unitgen.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\eu4unitgen.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\eu4unitgen.exe"; Description: "Start {#AppName} now (its setup wizard opens)"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Work: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    Work := ExpandConstant('{userdocs}\eu4unitgen');
    if DirExists(Work) then
      if MsgBox('Also delete your work in ' + Work + '? (references, projects, result sheets, exported mods)',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(Work, True, True, True);
  end;
end;
