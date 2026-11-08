; OneEighty Motion Blur - Inno Setup script (Inno Setup 6).
; Built by release/make_release.py, which passes /DAppVersion=x.y.z and the
; paths below. Manual build:
;   ISCC.exe /DAppVersion=0.6.0 release\OneEighty.iss

#ifndef AppVersion
  #error Pass /DAppVersion=x.y.z
#endif
#ifndef RootDir
  #define RootDir ".."
#endif

#define AppName "OneEighty Motion Blur"
#define Publisher "thinkvp"
#define RepoURL "https://github.com/thinkvp/OneEighty"
; Adobe's shared plug-in folder, scanned by Premiere Pro, After Effects and Media Encoder
#define PluginDir "{commonpf64}\Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty"

[Setup]
; Keep AppId fixed forever: it ties upgrades and the uninstaller together.
AppId={{6F1E3B8A-2C4D-4E9A-9B7C-1A80EE180180}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#Publisher}
AppPublisherURL={#RepoURL}
AppSupportURL={#RepoURL}/issues
AppUpdatesURL={#RepoURL}/releases
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoCompany={#Publisher}
DefaultDirName={autopf}\thinkvp\OneEighty
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=no
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#RootDir}\LICENSE
OutputDir={#RootDir}\dist
OutputBaseFilename=OneEighty-v{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\unins000.exe

[Files]
Source: "{#RootDir}\plugin\build\OneEighty.aex"; DestDir: "{#PluginDir}"; Flags: ignoreversion
; RIFE model for Quality "Extra High"
Source: "{#RootDir}\plugin\build\rife-v4.6\*"; DestDir: "{#PluginDir}\rife-v4.6"; Flags: ignoreversion
Source: "{#RootDir}\dist\README.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "{#RootDir}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#RootDir}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[UninstallDelete]
Type: dirifempty; Name: "{#PluginDir}\rife-v4.6"
Type: dirifempty; Name: "{#PluginDir}"

[Messages]
FinishedLabel=Setup has installed [name] on your computer.%n%nIn Premiere Pro: Effects > Video Effects > OneEighty > OneEighty Motion Blur.

[Code]
// FILE_ATTRIBUTE_REPARSE_POINT is predefined by Inno Setup.
const
  INVALID_FILE_ATTRIBUTES = $FFFFFFFF;

function GetFileAttributes(lpFileName: String): Cardinal;
  external 'GetFileAttributesW@kernel32.dll stdcall';

function IsProcessRunning(const ExeName: String): Boolean;
var
  Locator, Service, Procs: Variant;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\cimv2');
    Procs := Service.ExecQuery(Format('SELECT ProcessId FROM Win32_Process WHERE Name="%s"', [ExeName]));
    Result := Procs.Count > 0;
  except
    Result := False;
  end;
end;

function RunningAdobeApp(): String;
begin
  Result := '';
  if IsProcessRunning('Adobe Premiere Pro.exe') then Result := 'Premiere Pro'
  else if IsProcessRunning('AfterFX.exe') then Result := 'After Effects'
  else if IsProcessRunning('Adobe Media Encoder.exe') then Result := 'Media Encoder';
end;

// The plug-in file is locked while an Adobe app has it loaded.
function AdobeAppsClosed(const Action: String): Boolean;
var
  App: String;
begin
  Result := True;
  App := RunningAdobeApp();
  while App <> '' do
  begin
    if MsgBox('Please close ' + App + ' to ' + Action + ' {#AppName}.' + #13#10#13#10 +
              'Click Retry when it is closed, or Cancel to stop.', mbError, MB_RETRYCANCEL) = IDCANCEL then
    begin
      Result := False;
      exit;
    end;
    App := RunningAdobeApp();
  end;
end;

// A development setup links the plug-in folder to a build directory: leave it alone.
function PluginDirIsLink(): Boolean;
var
  Attr: Cardinal;
begin
  Attr := GetFileAttributes(ExpandConstant('{#PluginDir}'));
  Result := (Attr <> INVALID_FILE_ATTRIBUTES) and ((Attr and FILE_ATTRIBUTE_REPARSE_POINT) <> 0);
end;

function InitializeSetup(): Boolean;
begin
  Result := False;
  if PluginDirIsLink() then
  begin
    MsgBox(ExpandConstant('{#PluginDir}') + ' is a link to a development build.' + #13#10 +
           'Remove it first to install the release version.', mbError, MB_OK);
    exit;
  end;
  Result := AdobeAppsClosed('install');
end;

function InitializeUninstall(): Boolean;
begin
  Result := False;
  if PluginDirIsLink() then
  begin
    MsgBox(ExpandConstant('{#PluginDir}') + ' is a link to a development build; not uninstalling.', mbError, MB_OK);
    exit;
  end;
  Result := AdobeAppsClosed('uninstall');
end;
