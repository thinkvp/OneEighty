@echo off
rem Fetches the NVIDIA Optical Flow SDK headers (BSD branch). The runtime
rem (nvofapi64.dll) ships with the NVIDIA driver.
setlocal
cd /d "%~dp0"
if not exist nvof\.git (
    git clone --depth 1 -b nvof_2_0_bsd https://github.com/NVIDIA/NVIDIAOpticalFlowSDK.git nvof || exit /b 1
)
echo NVOF READY
