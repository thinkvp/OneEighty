@echo off
rem Installs OneEighty Motion Blur for Premiere Pro / After Effects (all users).
rem Copies OneEighty.aex into Adobe's shared plug-in folder. Needs administrator
rem rights (asks for them).
setlocal

net session >nul 2>&1
if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "SRC=%~dp0OneEighty.aex"
set "DEST=%ProgramFiles%\Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty"

if not exist "%SRC%" (
    echo OneEighty.aex was not found next to this script.
    goto :fail
)

for %%P in ("Adobe Premiere Pro.exe" "AfterFX.exe" "Adobe Media Encoder.exe") do (
    tasklist /fi "imagename eq %%~P" 2>nul | find /i "%%~P" >nul && (
        echo Please close %%~P first, then run this installer again.
        goto :fail
    )
)

rem A developer setup links this folder to a build directory: don't touch it.
fsutil reparsepoint query "%DEST%" >nul 2>&1 && (
    echo "%DEST%" is a link to a development build. Remove it first if you want the release version.
    goto :fail
)

if not exist "%DEST%" mkdir "%DEST%" || goto :fail
copy /y "%SRC%" "%DEST%\OneEighty.aex" >nul || goto :fail

echo.
echo OneEighty Motion Blur installed to:
echo   %DEST%
echo.
echo Start Premiere Pro and find it under Effects ^> Video Effects ^> OneEighty.
echo.
pause
exit /b 0

:fail
echo.
echo Installation did not complete.
pause
exit /b 1
