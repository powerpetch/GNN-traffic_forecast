@echo off
echo ========================================
echo    Running Model Evaluation
echo ========================================
echo.
echo Starting evaluation with confusion matrices...
echo.

python train.py

echo.
echo ========================================
echo    Evaluation Complete!
echo ========================================
echo.
echo Check outputs/ folder for:
echo   - confusion_matrices.png
echo   - training_history.png
echo   - evaluation_results.pkl
echo.
pause
