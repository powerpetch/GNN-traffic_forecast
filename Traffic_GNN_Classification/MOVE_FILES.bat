@echo off
echo ========================================
echo    Moving Files to Correct Location
echo ========================================
echo.
echo Moving confusion_matrices.png from old location to new location...
echo.

set OLD_PATH=D:\user\Data_project\Traffic_GNN_Classification\outputs
set NEW_PATH=D:\user\Data_project\Project_data\Traffic_GNN_Classification\outputs

if exist "%OLD_PATH%\confusion_matrices.png" (
    echo Found confusion_matrices.png in old location
    copy "%OLD_PATH%\confusion_matrices.png" "%NEW_PATH%\confusion_matrices.png"
    echo ✓ File copied successfully!
    echo.
    echo File location: %NEW_PATH%\confusion_matrices.png
) else (
    echo ❌ File not found in old location
    echo Generating new file in correct location...
    python test_confusion_matrix.py
)

echo.
echo ========================================
echo    Done!
echo ========================================
pause
