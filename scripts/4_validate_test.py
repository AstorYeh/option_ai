"""
Step 4: 測試集驗證 (含選擇權真實成本)
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import joblib
import json

# 重用優化腳本的函數
import importlib.util
spec = importlib.util.spec_from_file_location("optimize", "scripts/3_optimize_strategy.py")
optimize_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimize_module)
backtest_strategy = optimize_module.backtest_strategy
calculate_metrics = optimize_module.calculate_metrics

def validate_on_test_set():
    """在測試集上驗證策略 (含選擇權真實成本)"""
    logger.info("=== 測試集驗證 (含選擇權真實成本) ===")
    
    # 載入模型
    model = joblib.load("models/direction_model.pkl")
    
    # 載入最佳參數
    with open("results/optimization.json", 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    
    logger.info(f"使用最佳參數:")
    logger.info(f"  持有期間: {best_params['holding_days']:.0f} 天")
    logger.info(f"  信心度閾值: {best_params['confidence_threshold']:.2f}")
    logger.info(f"  停損點: {best_params['stop_loss']:.1%}")
    logger.info(f"  停利點: {best_params['take_profit']:.1%}")
    
    # 載入測試資料
    test_df = pd.read_csv("data/test/test_labeled.csv")
    logger.info(f"\n測試資料: {len(test_df)} 筆")
    logger.info(f"日期範圍: {test_df['date'].min()} ~ {test_df['date'].max()}")
    
    # 執行回測
    logger.info("\n執行測試集回測 (含 Delta/Theta/滑價)...")
    test_trades = backtest_strategy(test_df, model, best_params)
    
    # 計算績效
    test_metrics = calculate_metrics(test_trades)
    
    logger.info(f"\n=== 測試集績效 (含真實成本) ===")
    logger.info(f"總交易次數: {test_metrics['total_trades']:.0f}")
    logger.info(f"勝率: {test_metrics['win_rate']:.2%}")
    logger.info(f"平均報酬率: {test_metrics['avg_return']:.2%}")
    logger.info(f"總報酬率: {test_metrics['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    logger.info(f"Calmar Ratio: {test_metrics.get('calmar_ratio', 0):.4f}")
    logger.info(f"最大回撤: {test_metrics['max_drawdown']:.2%}")
    logger.info(f"Profit Factor: {test_metrics['profit_factor']:.4f}")
    logger.info(f"綜合評分: {test_metrics['score']:.4f}")
    
    # 與訓練集比較
    logger.info(f"\n=== 訓練集 vs 測試集 ===")
    logger.info(f"勝率: {best_params.get('win_rate', 0):.2%} -> {test_metrics['win_rate']:.2%}")
    logger.info(f"平均報酬: {best_params.get('avg_return', 0):.2%} -> {test_metrics['avg_return']:.2%}")
    logger.info(f"Sharpe: {best_params.get('sharpe_ratio', 0):.4f} -> {test_metrics['sharpe_ratio']:.4f}")
    logger.info(f"評分: {best_params.get('score', 0):.4f} -> {test_metrics['score']:.4f}")
    
    # 過擬合檢查
    train_score = best_params.get('score', 0)
    test_score = test_metrics['score']
    score_diff = abs(train_score - test_score)
    
    if score_diff > 0.3:
        logger.warning(f"[WARN] 嚴重過擬合! 評分差異: {score_diff:.4f}")
    elif score_diff > 0.15:
        logger.warning(f"[WARN] 輕微過擬合. 評分差異: {score_diff:.4f}")
    else:
        logger.info(f"[OK] 模型泛化良好,評分差異: {score_diff:.4f}")
    
    # 儲存結果
    results_dir = Path("results")
    
    # 儲存交易記錄
    test_trades.to_csv(results_dir / "backtest_test.csv", index=False)
    
    # 儲存測試集績效
    test_metrics['validation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    test_metrics['test_samples'] = len(test_df)
    test_metrics['overfitting_score'] = float(score_diff)
    test_metrics['train_score'] = float(train_score)
    
    with open(results_dir / "test_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[OK] 測試結果已儲存: {results_dir.absolute()}")
    
    # 策略評級
    score = test_metrics['score']
    if score >= 0.9:
        grade = "A+"
        recommendation = "優秀,可考慮實盤"
    elif score >= 0.8:
        grade = "A"
        recommendation = "良好,謹慎實盤"
    elif score >= 0.7:
        grade = "B"
        recommendation = "中等,需優化"
    elif score >= 0.6:
        grade = "C"
        recommendation = "不佳,需大幅改進"
    elif score >= 0.4:
        grade = "D"
        recommendation = "差,重新設計"
    else:
        grade = "F"
        recommendation = "失敗,根本性問題"
    
    logger.info(f"\n=== 策略評級 ===")
    logger.info(f"評級: {grade}")
    logger.info(f"建議: {recommendation}")
    
    return test_metrics, grade, recommendation

if __name__ == "__main__":
    try:
        test_metrics, grade, recommendation = validate_on_test_set()
        
        print("\n" + "="*60)
        print("[SUCCESS] 測試集驗證完成! (含選擇權真實成本)")
        print("="*60)
        print(f"\n測試集績效:")
        print(f"  勝率: {test_metrics['win_rate']:.2%}")
        print(f"  平均報酬: {test_metrics['avg_return']:.2%}")
        print(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
        print(f"  Calmar Ratio: {test_metrics.get('calmar_ratio', 0):.4f}")
        print(f"\n策略評級: {grade}")
        print(f"建議: {recommendation}")
        print("\n下一步: python scripts/5_generate_report.py")
        
    except Exception as e:
        logger.error(f"驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("[ERROR] 測試集驗證失敗!")
        print("="*60)

