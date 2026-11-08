@echo off
rem Fetches vcpkg and builds the plugin's static libraries: a minimal OpenCV
rem (core/imgproc/video) and ncnn with Vulkan (Quality "Extra High", RIFE).
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if not exist vcpkg\.git (
    git clone --depth 1 https://github.com/microsoft/vcpkg.git vcpkg || exit /b 1
)
if not exist vcpkg\vcpkg.exe (
    call vcpkg\bootstrap-vcpkg.bat -disableMetrics || exit /b 1
)
rem [core] = no default features (no codecs, dnn, etc.); x64-windows-static-md = static libs, /MD runtime
vcpkg\vcpkg.exe install "opencv4[core]:x64-windows-static-md" --disable-metrics || exit /b 1
rem ncnn loads the graphics driver's vulkan-1.dll at runtime (nothing extra to ship)
vcpkg\vcpkg.exe install "ncnn[vulkan]:x64-windows-static-md" --disable-metrics || exit /b 1
echo OPENCV + NCNN READY
