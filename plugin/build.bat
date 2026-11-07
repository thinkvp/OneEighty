@echo off
rem Builds build\OneEighty.aex (x64 Release) and build\oe_offline.exe with the VS 2022 Build Tools.
rem Needs ..\deps\setup_opencv.bat to have been run once (static OpenCV via vcpkg).
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem Adobe SDKs (not in git): sdk\ae = After Effects SDK 26.5, sdk\premiere = Premiere Pro 26.0 C++ SDK
set "SDK=%~dp0..\sdk\ae\Examples"
set "PRSDK=%~dp0..\sdk\premiere\Examples"
set "CV=%~dp0..\deps\vcpkg\installed\x64-windows-static-md"
set "OUT=%~dp0build"
set "OBJ=%~dp0build\obj"
if not exist "%OBJ%" mkdir "%OBJ%"

set CFLAGS=/nologo /c /O2 /EHsc /MD /W3 /WX /std:c++17 /DMSWindows /DWIN32 /D_WINDOWS /DNOMINMAX /I "%CV%\include\opencv4"
set CVLIBS="%CV%\lib\opencv_video4.lib" "%CV%\lib\opencv_imgproc4.lib" "%CV%\lib\opencv_core4.lib" "%CV%\lib\zs.lib"

echo [1/4] PiPL resource
cl /nologo /I "%SDK%\Headers" /EP OneEightyPiPL.r > "%OBJ%\OneEightyPiPL.rr" || exit /b 1
"%SDK%\Resources\PiPLtool.exe" "%OBJ%\OneEightyPiPL.rr" "%OBJ%\OneEightyPiPL.rrc" || exit /b 1
cl /nologo /D "MSWindows" /EP "%OBJ%\OneEightyPiPL.rrc" > "%OBJ%\OneEightyPiPL.rc" || exit /b 1
rc /nologo /fo "%OBJ%\OneEightyPiPL.res" "%OBJ%\OneEightyPiPL.rc" || exit /b 1

echo [2/4] Compile
cl %CFLAGS% /I "%SDK%\Headers" /I "%SDK%\Headers\SP" /I "%SDK%\Util" /I "%SDK%\Resources" ^
   /Fo"%OBJ%\\" OneEighty.cpp MotionBlur.cpp tools\oe_offline.cpp || exit /b 1

echo [3/4] Link offline test tool
link /nologo /OUT:"%OUT%\oe_offline.exe" "%OBJ%\oe_offline.obj" "%OBJ%\MotionBlur.obj" %CVLIBS% || exit /b 1

rem ---- GPU path (CUDA + NVIDIA Optical Flow); needs the CUDA toolkit and ..\deps\setup_nvof.bat
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "NVOF=%~dp0..\deps\nvof"
echo [GPU] Compile CUDA
"%CUDA_PATH%\bin\nvcc.exe" -c -O3 -std=c++17 -Xcompiler "/MD /EHsc /W3" -I "%NVOF%" ^
   -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 ^
   -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 ^
   -o "%OBJ%\GpuMotionBlur.obj" gpu\GpuMotionBlur.cu || exit /b 1
set CUDALIBS="%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cuda.lib"
cl %CFLAGS% /Fo"%OBJ%\\" tools\oe_gpu_offline.cpp || exit /b 1
echo [GPU] Link GPU test tool
link /nologo /OUT:"%OUT%\oe_gpu_offline.exe" "%OBJ%\oe_gpu_offline.obj" "%OBJ%\GpuMotionBlur.obj" "%OBJ%\MotionBlur.obj" ^
   %CVLIBS% %CUDALIBS% || exit /b 1

echo [4/4] Link plugin
link /nologo /DLL /OUT:"%OUT%\OneEighty.aex" /IMPLIB:"%OBJ%\OneEighty.lib" "%OBJ%\OneEighty.obj" "%OBJ%\MotionBlur.obj" ^
   "%OBJ%\OneEightyPiPL.res" %CVLIBS% kernel32.lib user32.lib || exit /b 1

echo BUILD OK: %OUT%\OneEighty.aex
