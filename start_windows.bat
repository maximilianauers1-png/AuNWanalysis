@echo off
REM  AuNW Analysis - start the program (double-click me)
REM  The program runs in this window and is used in the web browser.
REM  Keep this window open while working; close it to quit the program.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Please run install_windows.bat first.
    pause
    exit /b 1
)
echo.
echo  AuNW Analysis is starting ...
echo  Your browser will open http://localhost:8501 in a few seconds.
echo  KEEP THIS WINDOW OPEN while you work. Close it to quit the program.
echo.
REM open the browser after a short delay, when the server is ready
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app.py
pause
