@echo off
echo ========================================
echo    Testing Congestion Distribution
echo ========================================
echo.
echo Testing how congestion levels change by time...
echo.

cd /d D:\user\Data_project\Project_data\Traffic_GNN_Classification

python test_congestion_distribution.py

echo.
echo ========================================
echo    Test Complete
echo ========================================
echo.
echo Check if distributions make sense:
echo   - Night time should be mostly Free-flow
echo   - Rush hours should have more Gridlock/Congested
echo.
pause
