@echo off
rem Builds build\OneEighty.aex (x64 Release) plus the offline test tools, with
rem the VS 2022 Build Tools and the CUDA toolkit.
rem One-time setup: ..\deps\setup_opencv.bat and ..\deps\setup_nvof.bat.
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem Adobe SDKs (not in git): sdk\ae = After Effects SDK 26.5, sdk\premiere = Premiere Pro 26.0 C++ SDK
set "SDK=%~dp0..\sdk\ae\Examples"
set "PRSDK=%~dp0..\sdk\premiere\Examples"
set "CV=%~dp0..\deps\vcpkg\installed\x64-windows-static-md"
set "NVOF=%~dp0..\deps\nvof"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "OUT=%~dp0build"
set "OBJ=%~dp0build\obj"
if not exist "%OBJ%" mkdir "%OBJ%"

set CFLAGS=/nologo /c /O2 /EHsc /MD /W3 /WX /std:c++17 /DMSWindows /DWIN32 /D_WINDOWS /DNOMINMAX /I "%CV%\include\opencv4"
set AEINC=/I "%SDK%\Headers" /I "%SDK%\Headers\SP" /I "%SDK%\Util" /I "%SDK%\Resources"
set CVLIBS="%CV%\lib\opencv_video4.lib" "%CV%\lib\opencv_imgproc4.lib" "%CV%\lib\opencv_core4.lib" "%CV%\lib\zs.lib"
set CUDALIBS="%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cuda.lib"

echo [1/5] PiPL resource
cl /nologo /I "%SDK%\Headers" /EP OneEightyPiPL.r > "%OBJ%\OneEightyPiPL.rr" || exit /b 1
"%SDK%\Resources\PiPLtool.exe" "%OBJ%\OneEightyPiPL.rr" "%OBJ%\OneEightyPiPL.rrc" || exit /b 1
cl /nologo /D "MSWindows" /EP "%OBJ%\OneEightyPiPL.rrc" > "%OBJ%\OneEightyPiPL.rc" || exit /b 1
rc /nologo /fo "%OBJ%\OneEightyPiPL.res" "%OBJ%\OneEightyPiPL.rc" || exit /b 1

echo [2/5] Compile C++
cl %CFLAGS% %AEINC% /Fo"%OBJ%\\" OneEighty.cpp MotionBlur.cpp Log.cpp tools\oe_offline.cpp tools\oe_gpu_offline.cpp || exit /b 1

echo [3/5] Compile CUDA
"%CUDA_PATH%\bin\nvcc.exe" -c -O3 -std=c++17 -Xcompiler "/MD /EHsc /W3" -I "%NVOF%" ^
   -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 ^
   -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 ^
   -o "%OBJ%\GpuMotionBlur.obj" gpu\GpuMotionBlur.cu || exit /b 1

echo [4/5] Link offline test tools
link /nologo /OUT:"%OUT%\oe_offline.exe" "%OBJ%\oe_offline.obj" "%OBJ%\MotionBlur.obj" %CVLIBS% || exit /b 1
link /nologo /OUT:"%OUT%\oe_gpu_offline.exe" "%OBJ%\oe_gpu_offline.obj" "%OBJ%\GpuMotionBlur.obj" "%OBJ%\MotionBlur.obj" ^
   %CVLIBS% %CUDALIBS% || exit /b 1

echo [5/5] Link plugin
rem CUDA 13's cuda.lib/cudart_static load the NVIDIA driver at runtime (no nvcuda.dll import),
rem so the plugin still loads, CPU-only, on machines without NVIDIA.
link /nologo /DLL /OUT:"%OUT%\OneEighty.aex" /IMPLIB:"%OBJ%\OneEighty.lib" ^
   "%OBJ%\OneEighty.obj" "%OBJ%\MotionBlur.obj" "%OBJ%\Log.obj" "%OBJ%\GpuMotionBlur.obj" ^
   "%OBJ%\OneEightyPiPL.res" %CVLIBS% %CUDALIBS% kernel32.lib user32.lib || exit /b 1

echo BUILD OK: %OUT%\OneEighty.aex
