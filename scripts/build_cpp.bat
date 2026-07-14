@echo off
REM 在 x64 Native Tools Command Prompt for VS 2026 中运行此脚本
call "D:\SDK\VC\Auxiliary\Build\vcvars64.bat"
conda activate agentrag
cd /d "D:\010\git工作区\agentRAG项目"
if exist build rmdir /s /q build
mkdir build
cd build
cmake .. -G "NMake Makefiles"
nmake
echo.
echo Build complete! Check for agentrag_core*.pyd in build/
dir *.pyd 2>nul
