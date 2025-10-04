@echo off
REM ========================================
REM  Traffic GNN Classification Dashboard
REM  Bangkok Traffic Analysis System
REM ========================================

echo.
echo ========================================
echo   Starting Traffic GNN Dashboard
echo ========================================
echo.

cd /d "%~dp0"

echo Starting Streamlit dashboard...
echo.
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

py -m streamlit run app/dashboard.py

pause
