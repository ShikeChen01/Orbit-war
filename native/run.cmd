@echo off
REM Run a native exe (ow_train / ow_eval) with LibTorch's DLLs on PATH.
REM The exes load-time-link torch_cpu/torch_cuda/c10 from the wheel's torch\lib, so that
REM dir must be discoverable before main() runs. Usage (from repo root):
REM   native\run.cmd ow_train --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp --total-steps 200000 --out runs/native/test.owc
REM   native\run.cmd ow_eval  --ckpt runs/native/test.owc --worlds runs/native/eval.owp
setlocal
set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" -c "import torch,os;print(os.path.dirname(torch.__file__))" > "%TEMP%\ow_troot.txt"
set /p TORCH_ROOT=<"%TEMP%\ow_troot.txt"
set "PATH=%TORCH_ROOT%\lib;%PATH%"
set "EXE=%~dp0_bld\%~1.exe"
if not exist "%EXE%" ( echo not built: %EXE% ^(run native\build.cmd first^) & exit /b 1 )
REM Forward every arg after the exe name (%%b = everything past token 1).
set "ALL=%*"
for /f "tokens=1,*" %%a in ("%ALL%") do "%EXE%" %%b
