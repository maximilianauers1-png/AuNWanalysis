@echo off
REM  AuNW Analysis - start the program (double-click me)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Please run install_windows.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
