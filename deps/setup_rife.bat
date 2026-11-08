@echo off
rem Fetches the RIFE v4.6 model (ncnn format, MIT) used by Quality "Extra High"
rem into deps\rife\rife-v4.6 (build.bat copies it next to the plugin).
rem Source: github.com/nihui/rife-ncnn-vulkan, pinned commit, checked by SHA-256.
setlocal
cd /d "%~dp0"
set "COMMIT=5ca8c773f2e9ae78ba971bb24c35c1bad5fbaf31"
set "URL=https://raw.githubusercontent.com/nihui/rife-ncnn-vulkan/%COMMIT%/models/rife-v4.6"
set "DST=rife\rife-v4.6"
set "BIN_SHA256=f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958"
if not exist "%DST%" mkdir "%DST%"
curl.exe -fsSL -o "%DST%\flownet.param" "%URL%/flownet.param" || exit /b 1
curl.exe -fsSL -o "%DST%\flownet.bin" "%URL%/flownet.bin" || exit /b 1
for /f "tokens=* delims=" %%h in ('certutil -hashfile "%DST%\flownet.bin" SHA256 ^| findstr /v ":"') do (
    if /i not "%%h"=="%BIN_SHA256%" (
        echo flownet.bin SHA-256 mismatch: %%h
        exit /b 1
    )
)
echo RIFE MODEL READY
