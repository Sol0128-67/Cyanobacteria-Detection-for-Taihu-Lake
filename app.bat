@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo   太湖蓝藻监测软件 (ExG 热力图分析)
echo   把影像文件直接拖进软件窗口即可分析
echo ================================================
".venv\Scripts\python.exe" app.py
pause

