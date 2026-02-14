# 自我學習訓練 - Windows 批次執行腳本

@echo off
echo ==================================
echo 台指期選擇權 - 自我學習訓練
echo ==================================
echo.

REM Step 1: 資料準備
echo [1/5] 資料準備與分割...
python scripts\1_prepare_data.py
if errorlevel 1 (
    echo ❌ 資料準備失敗!
    exit /b 1
)
echo.

REM Step 2: 模型訓練
echo [2/5] 模型訓練...
python scripts\2_train_models.py
if errorlevel 1 (
    echo ❌ 模型訓練失敗!
    exit /b 1
)
echo.

REM Step 3: 策略優化
echo [3/5] 策略參數優化...
python scripts\3_optimize_strategy.py
if errorlevel 1 (
    echo ❌ 策略優化失敗!
    exit /b 1
)
echo.

REM Step 4: 測試驗證
echo [4/5] 測試集驗證...
python scripts\4_validate_test.py
if errorlevel 1 (
    echo ❌ 測試驗證失敗!
    exit /b 1
)
echo.

REM Step 5: 生成報告
echo [5/5] 生成訓練報告...
python scripts\5_generate_report.py
if errorlevel 1 (
    echo ❌ 報告生成失敗!
    exit /b 1
)
echo.

echo ==================================
echo ✅ 自我學習訓練完成!
echo ==================================
echo.
echo 查看報告: results\training_report.md
pause
