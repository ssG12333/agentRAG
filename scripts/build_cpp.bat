@echo off
setlocal
chcp 65001 >nul

set "ENV_ROOT=E:\anaconda3\.conda\envs\agentrag"
set "PROJECT_ROOT=%~dp0.."
set "CMAKE=%ENV_ROOT%\Scripts\cmake.exe"
set "PYTHON=%ENV_ROOT%\python.exe"
set "PYBIND11_DIR=%ENV_ROOT%\Lib\site-packages\pybind11\share\cmake\pybind11"
set "PATH=%ENV_ROOT%;%ENV_ROOT%\Scripts;%ENV_ROOT%\Library\bin;%PATH%"

call "D:\SDK\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

if not exist "%CMAKE%" (
    echo ERROR: CMake not found: %CMAKE%
    exit /b 1
)
if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    exit /b 1
)
if not exist "%PYBIND11_DIR%\pybind11Config.cmake" (
    echo ERROR: pybind11 CMake package not found: %PYBIND11_DIR%
    exit /b 1
)

pushd "%PROJECT_ROOT%"

"%CMAKE%" -S . -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -Dpybind11_DIR="%PYBIND11_DIR%" ^
    -DPython_EXECUTABLE="%PYTHON%"
if errorlevel 1 goto :fail

"%CMAKE%" --build build --config Release
if errorlevel 1 goto :fail

echo.
echo Build complete! Extension copied to project root:
dir /b agentrag_core*.pyd 2>nul
popd
exit /b 0

:fail
popd
exit /b 1
