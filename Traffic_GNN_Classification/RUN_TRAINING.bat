@echo off
REM ========================================
REM  Traffic GNN Classification Training
REM  Model Training Script
REM ========================================

echo.
echo ========================================
echo   Training Traffic GNN Model
echo ========================================
echo.

cd /d "%~dp0"

echo Starting ENHANCED model training...
echo.
echo Improvements Applied:
echo   - AdamW optimizer (better weight decay)
echo   - Learning rate scheduling (auto-reduce on plateau)
echo   - Early stopping (stops if no improvement for 20 epochs)
echo   - Gradient clipping (prevents training instability)
echo   - Data augmentation (noise injection for robustness)
echo.
echo This may take 10-30 minutes depending on your hardware
echo Training will auto-stop if model stops improving!
echo.

py train.py --epochs 100 --batch_size 32 --patience 20

echo.
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo Check the 'outputs' folder for:
echo   - Trained model files (*.pth)
echo   - Training history plots
echo   - Performance metrics
echo.

pause
