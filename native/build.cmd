@echo off
REM Build the native module with MSVC + Ninja against the venv's nanobind + LibTorch.
REM Run from repo root:  cmd /c native\build.cmd
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

set "PATH=%~dp0..\.venv\Scripts;%PATH%"
set "PY=%~dp0..\.venv\Scripts\python.exe"

REM NB_DIR and TORCH_DIR are expected from the environment (set by the caller).
REM Fall back to computing them via a temp file if not provided.
if "%NB_DIR%"=="" (
  "%PY%" -c "import nanobind;print(nanobind.cmake_dir())" > "%TEMP%\nbdir.txt"
  set /p NB_DIR=<"%TEMP%\nbdir.txt"
)
if "%TORCH_ROOT%"=="" (
  "%PY%" -c "import torch,os;print(os.path.dirname(torch.__file__))" > "%TEMP%\troot.txt"
  set /p TORCH_ROOT=<"%TEMP%\troot.txt"
)

echo nanobind_DIR=%NB_DIR%
echo TORCH_INSTALL_DIR=%TORCH_ROOT%

cmake -G Ninja -S "%~dp0." -B "%~dp0_bld" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython_EXECUTABLE="%PY%" ^
  -Dnanobind_DIR="%NB_DIR%" ^
  -DTORCH_INSTALL_DIR="%TORCH_ROOT%" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
if errorlevel 1 ( echo configure failed & exit /b 1 )

cmake --build "%~dp0_bld" --config Release
if errorlevel 1 ( echo build failed & exit /b 1 )

echo BUILD_DONE
