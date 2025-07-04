; -- LadbonAIDesktop.iss --
; Inno Setup script for Ladbon AI Desktop (fixed & compilable)

#define MyAppName      "Ladbon AI Desktop"
#define MyAppVersion   "1.1"
#define MyAppExeName   "Ladbon AI Desktop.exe"
#define IconSource     "ladbon_ai.ico"
#define RedistExe      "VC_redist.x64.exe"   ; put this in a "redist" subfolder
#define AppDataName    "Ladbon AI Desktop"   ; goes under %LOCALAPPDATA%

#pragma message "Checking for icon file..."
#if FileExists(AddBackslash(SourcePath) + IconSource)
  #define USEICONFILE
  #pragma message "Icon file found!"
#else
  #pragma message "Icon file not found."
#endif

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf64}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
DiskSpanning=yes
SlicesPerDisk=1
OutputDir=installer
OutputBaseFilename=Ladbon_AI_Desktop_Setup

; Force 64-bit mode
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
; Always install to Program Files (never Program Files (x86))
DisableProgramGroupPage=yes
UsePreviousAppDir=no
AlwaysShowDirOnReadyPage=yes
#ifdef USEICONFILE
SetupIconFile={#AddBackslash(SourcePath) + IconSource}
#endif

[Dirs]
Name: "{app}"
Name: "{localappdata}\{#AppDataName}"
Name: "{localappdata}\{#AppDataName}\docs"
Name: "{localappdata}\{#AppDataName}\img"
Name: "{localappdata}\{#AppDataName}\logs"
Name: "{localappdata}\{#AppDataName}\models"

[Files]
; 1) Core app - includes all files and folders from the PyInstaller build
Source: "dist\{#MyAppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 2) VC++ redist (to {tmp})
Source: "redist\{#RedistExe}"; DestDir: "{tmp}"; Flags: deleteafterinstall
; 3) PATH fixing batch file
Source: "Launch_With_Correct_PATH.bat"; DestDir: "{app}"; Flags: ignoreversion
; 4) Data folders
Source: "docs\*";   DestDir: "{localappdata}\{#AppDataName}\docs";   Flags: ignoreversion recursesubdirs createallsubdirs
Source: "img\*";    DestDir: "{localappdata}\{#AppDataName}\img";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\*"; DestDir: "{localappdata}\{#AppDataName}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "settings.json"; DestDir: "{localappdata}\{#AppDataName}"; Flags: ignoreversion
; 5) Empty log placeholder
Source: "logs\README.txt"; DestDir: "{localappdata}\{#AppDataName}\logs"; Flags: ignoreversion skipifsourcedoesntexist
; 6) Icon
#ifdef USEICONFILE
Source: "{#AddBackslash(SourcePath) + IconSource}"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\{#MyAppName}";      Filename: "{app}\Launch_With_Correct_PATH.bat"; IconFilename: "{app}\{#IconSource}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\Launch_With_Correct_PATH.bat"; IconFilename: "{app}\{#IconSource}"

[Run]
; 1) Install VC++ runtime silently
Filename: "{tmp}\{#RedistExe}"; Parameters: "/quiet /norestart"; \
  StatusMsg: "Installing Visual C++ Runtime…"; Flags: runhidden

; 2) Launch the app with correct PATH (using the batch file)
Filename: "{app}\Launch_With_Correct_PATH.bat"; Description: "Launch {#MyAppName}"; \
  WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent unchecked

[Messages]
FinishedLabel=Setup has finished installing {#MyAppName}. \
Models live in %LOCALAPPDATA%\{#AppDataName}\models. \
Drop extra GGUFs there whenever you like. For Ollama support, grab it from ollama.com/download.

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('This installer will set up ' + '{#MyAppName}' + '.' + #13#10#13#10 +
         'Local models belong in "%LOCALAPPDATA%\{#AppDataName}\models".' + #13#10 +
         'Need more models? Drop GGUF files there after install.' + #13#10#13#10 +
         'Want Ollama functionality? Install it separately from ollama.com/download.',
         mbInformation, MB_OK);
end;
