```inno
; -- LadboAIDesktop.iss --
; Inno Setup script for Ladbon AI Desktop

[Setup]
AppName=Ladbon AI Desktop
AppVersion=1.0
DefaultDirName={autopf}\Ladbon AI Desktop
DefaultGroupName=Ladbon AI Desktop
UninstallDisplayIcon={app}\Ladbon AI Desktop.exe
Compression=lzma2
SolidCompression=yes
OutputDir=installer
OutputBaseFilename=Ladbon_AI_Desktop_Setup
SetupIconFile=app_icon.ico

[Dirs]
Name: "{app}\docs"; Permissions: users-modify
Name: "{app}\img"; Permissions: users-modify
Name: "{app}\logs"; Permissions: users-modify

[Files]
Source: "dist\Ladbon AI Desktop.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "img\*"; DestDir: "{app}\img"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "settings.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Ladbon AI Desktop"; Filename: "{app}\Ladbon AI Desktop.exe"
Name: "{commondesktop}\Ladbon AI Desktop"; Filename: "{app}\Ladbon AI Desktop.exe"

[Run]
Filename: "{app}\Ladbon AI Desktop.exe"; Description: "Launch Ladbon AI Desktop"; Flags: nowait postinstall skipifsilent

[Messages]
FinishedLabel=Setup has finished installing Ladbon AI Desktop on your computer. Remember that you need to install Ollama separately from ollama.com/download.

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('This installer will set up Ladbon AI Desktop.' + #13#10 + #13#10 +
         'Please note that Ollama must be installed separately from ollama.com/download', mbInformation, MB_OK);
end;
```