@echo off
setlocal
set "ROOT=%~dp0.."
python "%ROOT%\app\gesture_monitor.py"
if errorlevel 1 pause
endlocal
