@echo off
cd /d "D:\Users\PPT Master\battery_safety_agent"
echo ============================================
echo   Battery Safety Dashboard - starting...
echo   Keep this window OPEN. Closing it stops.
echo   Your browser will open automatically.
echo ============================================
".venv\Scripts\streamlit.exe" run app.py
pause
