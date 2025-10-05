@echo off
echo ========================================
echo    Model Detection Test
echo ========================================
echo.
echo Testing if models can be detected...
echo.

cd /d D:\user\Data_project\Project_data\Traffic_GNN_Classification

python test_model_detection.py

echo.
echo ========================================
echo    Test Complete
echo ========================================
echo.
echo If models were found, restart the dashboard:
echo    streamlit run app/dashboard.py
echo.
pause
