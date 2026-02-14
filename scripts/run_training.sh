#!/bin/bash
# 自我學習訓練 - 完整執行腳本

echo "=================================="
echo "台指期選擇權 - 自我學習訓練"
echo "=================================="
echo ""

# Step 1: 資料準備
echo "[1/5] 資料準備與分割..."
python scripts/1_prepare_data.py
if [ $? -ne 0 ]; then
    echo "❌ 資料準備失敗!"
    exit 1
fi
echo ""

# Step 2: 模型訓練
echo "[2/5] 模型訓練..."
python scripts/2_train_models.py
if [ $? -ne 0 ]; then
    echo "❌ 模型訓練失敗!"
    exit 1
fi
echo ""

# Step 3: 策略優化
echo "[3/5] 策略參數優化..."
python scripts/3_optimize_strategy.py
if [ $? -ne 0 ]; then
    echo "❌ 策略優化失敗!"
    exit 1
fi
echo ""

# Step 4: 測試驗證
echo "[4/5] 測試集驗證..."
python scripts/4_validate_test.py
if [ $? -ne 0 ]; then
    echo "❌ 測試驗證失敗!"
    exit 1
fi
echo ""

# Step 5: 生成報告
echo "[5/5] 生成訓練報告..."
python scripts/5_generate_report.py
if [ $? -ne 0 ]; then
    echo "❌ 報告生成失敗!"
    exit 1
fi
echo ""

echo "=================================="
echo "✅ 自我學習訓練完成!"
echo "=================================="
echo ""
echo "查看報告: results/training_report.md"
