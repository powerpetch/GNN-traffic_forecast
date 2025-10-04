@echo off
REM ========================================
REM  Enhanced GNN Training Script
REM  Advanced Training with Better Models
REM ========================================

echo.
echo ========================================
echo   Enhanced GNN Training
echo ========================================
echo.

cd /d "%~dp0"

echo Training enhanced model with:
echo   - Deeper network (128 hidden units)
echo   - Batch normalization
echo   - Learning rate scheduling
echo   - Early stopping
echo   - Data augmentation
echo.

py enhanced_train.py --epochs 150 --batch_size 64 --hidden_dim 128 --dropout 0.3 --lr 0.001 --patience 20

echo.
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo Check 'outputs' folder for:
echo   - best_enhanced_model.pth
echo   - enhanced_training_history.png
echo   - enhanced_confusion_matrices.png
echo.

pause
