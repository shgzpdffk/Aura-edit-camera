@echo off
cd /d "%~dp0"
"runtime\python.exe" -X utf8 "app\diagnostics.py"
pause
