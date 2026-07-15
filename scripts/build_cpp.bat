@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0.."

if defined AGENTRAG_PYTHON (
    set "PYTHON=%AGENTRAG_PYTHON%"
) else (
    for /f "delims=" %%I in ('where python 2^>nul') do if not defined PYTHON set "PYTHON=%%I"
)
if not defined PYTHON (
    echo ERROR: Python not found. Activate the project environment or set AGENTRAG_PYTHON.
    exit /b 1
)

if defined AGENTRAG_CMAKE (
    set "CMAKE=%AGENTRAG_CMAKE%"
) else (
    for /f "delims=" %%I in ('where cmake 2^>nul') do if not defined CMAKE set "CMAKE=%%I"
)
if not defined CMAKE (
    echo ERROR: CMake not found. Install CMake or set AGENTRAG_CMAKE.
    exit /b 1
)

for /f "delims=" %%I in ('"%PYTHON%" -m pybind11 --cmakedir 2^>nul') do if not defined PYBIND11_DIR set "PYBIND11_DIR=%%I"
if not exist "%PYBIND11_DIR%\pybind11Config.cmake" (
    echo ERROR: pybind11 not found for %PYTHON%. Run: "%PYTHON%" -m pip install pybind11
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    if defined AGENTRAG_VCVARS if exist "%AGENTRAG_VCVARS%" call "%AGENTRAG_VCVARS%"
)
where cl >nul 2>nul
if errorlevel 1 call :find_msvc
where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: MSVC compiler not found. Use a Developer Command Prompt or set AGENTRAG_VCVARS.
    exit /b 1
)

set "GENERATOR="
where ninja >nul 2>nul
if not errorlevel 1 set "GENERATOR=Ninja"

pushd "%PROJECT_ROOT%"

if defined GENERATOR (
    "%CMAKE%" -S . -B build -G "%GENERATOR%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="%PYBIND11_DIR%" ^
        -DPython_EXECUTABLE="%PYTHON%"
) else (
    "%CMAKE%" -S . -B build ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="%PYBIND11_DIR%" ^
        -DPython_EXECUTABLE="%PYTHON%"
)
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

:find_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do if not defined VS_ROOT set "VS_ROOT=%%I"
if defined VS_ROOT call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
exit /b 0
