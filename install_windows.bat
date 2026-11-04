@echo off
REM ---------------------------------------------------------------------
REM  AuNW Analysis - one-time installation on Windows (double-click me)
REM  Uses the official Python from python.org via the "py" launcher,
REM  creates a virtual environment in .venv and installs all packages.
REM ---------------------------------------------------------------------
cd /d "%~dp0"

set PYVER=
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PYVER (
        py -%%V -c "import sys" >nul 2>&1 && set PYVER=%%V
    )
)
if not defined PYVER (
    echo.
    echo  No suitable Python found. Please install Python 3.12 from
    echo  https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
    echo  and run this file again.
    echo.
    pause
    exit /b 1
)
echo Using Python %PYVER%

if exist .venv rmdir /s /q .venv
py -%PYVER% -m venv .venv
if errorlevel 1 ( echo Creating the virtual environment failed. & pause & exit /b 1 )

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo Installing the packages failed. & pause & exit /b 1 )

REM native application window (optional: without it the program opens in Edge/Chrome app mode)
.venv\Scripts\python.exe -m pip install pywebview
if errorlevel 1 echo pywebview could not be installed - the app will open in an Edge/Chrome window instead.

REM desktop + start-menu shortcut with the beaker icon
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\create_shortcut.ps1"

echo.
echo  Installation finished. Start the program with the "AuNW Analysis" icon on your desktop.
echo.
pause
