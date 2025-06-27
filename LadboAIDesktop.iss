; -- LadboAIDesktop.iss --
; Inno Setup script for Ladbon AI Desktop

#define MyAppName "Ladbon AI Desktop"
#define MyAppVersion "1.1"
#define MyAppExeName "Ladbon AI Desktop.exe"
#define IconSource "ladbon_ai.ico"

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
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
OutputDir=installer
OutputBaseFilename=Ladbon_AI_Desktop_Setup
; Set icon if available
#ifdef USEICONFILE
SetupIconFile={#AddBackslash(SourcePath) + IconSource}
#endif

[Dirs]
Name: "{app}\docs"; Permissions: users-modify
Name: "{app}\img"; Permissions: users-modify
Name: "{app}\logs"; Permissions: users-modify
Name: "{app}\models"; Permissions: users-modify

[Files]
Source: "dist\{#MyAppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "img\*"; DestDir: "{app}\img"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "settings.json"; DestDir: "{app}"; Flags: ignoreversion
#ifdef USEICONFILE
Source: "{#AddBackslash(SourcePath) + IconSource}"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Messages]
FinishedLabel=Setup has finished installing {#MyAppName} on your computer. You can place GGUF model files in the 'models' folder for local LLM support. Optionally, you can install Ollama separately from ollama.com/download for additional model options.

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('This installer will set up ' + '{#MyAppName}' + '.' + #13#10 + #13#10 +
         'For local AI models: Place GGUF model files in the "models" folder after installation.' + #13#10 + #13#10 +
         'For Ollama models: Install Ollama separately from ollama.com/download', mbInformation, MB_OK);
end;