@echo off
REM Windows launcher for myvideoai CLI.
REM Resolves the project root from this script's own location, adds the
REM vendored wheels (if present) to PYTHONPATH, then runs the package.
setlocal
REM project root = parent directory of the myvideoai\ package folder
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VENDORED="
if exist "%ROOT%\.vendored\torch_pkgs" set "VENDORED=%VENDORED;%ROOT%\.vendored\torch_pkgs"
if exist "%ROOT%\.vendored\py_pkgs" set "VENDORED=%VENDORED;%ROOT%\.vendored\py_pkgs"
if defined VENDORED set "PYTHONPATH=%VENDORED:~1%;%PYTHONPATH%"
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%ROOT%"
python -m myvideoai %*
endlocal
