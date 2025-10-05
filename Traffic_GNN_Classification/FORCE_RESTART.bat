@echo off
echo ========================================
echo    FORCE RESTART Dashboard
echo ========================================
echo.
echo This will:
echo   1. Kill all Streamlit processes
echo   2. Clear Python cache (__pycache__)
echo   3. Clear Streamlit cache
echo   4. Restart Dashboard
echo.
pause

cd /d D:\user\Data_project\Project_data\Traffic_GNN_Classification

echo.
echo [1/4] Killing Streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 >nul

echo [2/4] Clearing Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

echo [3/4] Clearing Streamlit cache...
if exist .streamlit\cache rd /s /q .streamlit\cache

echo [4/4] Starting Dashboard...
echo.
echo ========================================
echo    Dashboard Starting...
echo ========================================
echo.
echo IMPORTANT:
echo   1. Open in Incognito/Private Mode
echo   2. Use Ctrl+Shift+R to hard refresh
echo   3. Look for Debug Info in GNN Graph tab
echo.

streamlit run app/dashboard.py

pause
