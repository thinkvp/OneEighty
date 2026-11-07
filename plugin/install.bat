@echo off
rem One-time install (run as administrator): links Premiere/AE's shared plug-in
rem folder to build\, so every rebuild is picked up on the next app restart.
set "DEST=C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty"
if exist "%DEST%" (
    echo Already installed: %DEST%
    exit /b 0
)
mklink /J "%DEST%" "%~dp0build"
