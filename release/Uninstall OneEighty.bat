@echo off
rem Removes OneEighty Motion Blur from Adobe's shared plug-in folder (asks for
rem administrator rights).
setlocal

net session >nul 2>&1
if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "DEST=%ProgramFiles%\Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty"

for %%P in ("Adobe Premiere Pro.exe" "AfterFX.exe" "Adobe Media Encoder.exe") do (
    tasklist /fi "imagename eq %%~P" 2>nul | find /i "%%~P" >nul && (
        echo Please close %%~P first, then run this again.
        pause
        exit /b 1
    )
)

fsutil reparsepoint query "%DEST%" >nul 2>&1 && (
    echo "%DEST%" is a link to a development build; leaving it alone.
    pause
    exit /b 1
)

if exist "%DEST%\OneEighty.aex" del /f "%DEST%\OneEighty.aex"
if exist "%DEST%\rife-v4.6" rmdir /s /q "%DEST%\rife-v4.6"
if exist "%DEST%" rmdir "%DEST%" 2>nul

echo OneEighty Motion Blur removed.
pause
