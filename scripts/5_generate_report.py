"""
Step 5: 生成訓練報告 (增強版)
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime
from loguru import logger

def generate_report():
    """生成完整訓練報告 (增強版)"""
    logger.info("=== 生成訓練報告 ===")
    
    # 載入所有結果
    with open("data/data_summary.json", 'r', encoding='utf-8') as f:
        data_summary = json.load(f)
    
    with open("models/direction_model_metadata.json", 'r', encoding='utf-8') as f:
        model_metadata = json.load(f)
    
    with open("results/optimization.json", 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    
    with open("results/test_metrics.json", 'r', encoding='utf-8') as f:
        test_metrics = json.load(f)
    
    # 載入交易記錄
    train_trades = pd.read_csv("results/backtest_train.csv")
    test_trades = pd.read_csv("results/backtest_test.csv")
    
    # 評級
    score = test_metrics['score']
    if score >= 0.9:
        grade = "A+"
    elif score >= 0.8:
        grade = "A"
    elif score >= 0.7:
        grade = "B"
    elif score >= 0.6:
        grade = "C"
    elif score >= 0.4:
        grade = "D"
    else:
        grade = "F"
    
    # 安全取值輔助函數
    def safe_get(d, key, default=0):
        v = d.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    
    # 生成 Markdown 報告
    report = f"""# 台指期選擇權 - 自我學習訓練報告

**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**策略評級**: **{grade}**  
**測試集評分**: {test_metrics['score']:.4f}

---

## [DATA] 資料統計

### 資料分割
- **總資料筆數**: {data_summary['total_records']}
- **訓練集**: {data_summary['train_records']} 筆 ({data_summary['train_records']/data_summary['total_records']*100:.1f}%)
  - 日期範圍: {data_summary['train_date_range']}
- **測試集**: {data_summary['test_records']} 筆 ({data_summary['test_records']/data_summary['total_records']*100:.1f}%)
  - 日期範圍: {data_summary['test_date_range']}

### 特徵工程
- **特徵數量**: {model_metadata['num_features']}
- **特徵類型**: 技術指標 + 價格衍生特徵 (排除原始 OHLCV)
- **已排除**: {', '.join(model_metadata.get('excluded_price_cols', ['open', 'high', 'low', 'close', 'volume']))}

---

## [MODEL] 模型訓練

### 方向性預測模型
- **模型類型**: {model_metadata.get('model_type', 'XGBClassifier')}
- **交叉驗證**: {model_metadata.get('cv_type', 'KFold')} ({model_metadata.get('cv_folds', 3)} 折)
- **訓練時間**: {model_metadata['train_date']}
- **訓練樣本**: {model_metadata['train_samples']} 筆

### 過擬合三方對比

| 指標 | 訓練集 | 交叉驗證 | 測試集 |
|------|--------|----------|--------|
| 準確率/評分 | {model_metadata['train_accuracy']:.2%} | {model_metadata['cv_score']:.2%} | {test_metrics['score']:.4f} |

- **過擬合差距 (訓練-CV)**: {model_metadata.get('overfit_gap', model_metadata['train_accuracy'] - model_metadata['cv_score']):.2%}
"""
    
    overfit_gap = model_metadata.get('overfit_gap', model_metadata['train_accuracy'] - model_metadata['cv_score'])
    if overfit_gap > 0.2:
        report += "- **結論**: !! 嚴重過擬合, 需繼續改進\n"
    elif overfit_gap > 0.1:
        report += "- **結論**: ! 輕微過擬合, 可接受\n"
    else:
        report += "- **結論**: OK 泛化良好\n"

    report += f"""
### 最佳超參數
```json
{json.dumps(model_metadata['best_params'], indent=2, ensure_ascii=False)}
```

### Top 10 重要特徵
"""
    
    for i, feat in enumerate(model_metadata['feature_importance'][:10], 1):
        report += f"{i}. **{feat['feature']}**: {feat['importance']:.4f}\n"
    
    # 特徵洩漏檢查
    raw_price_cols = ['open', 'high', 'low', 'close', 'volume']
    leak_features = [f['feature'] for f in model_metadata['feature_importance'][:5] 
                     if f['feature'] in raw_price_cols]
    if leak_features:
        report += f"\n! 原始價格特徵洩漏: {leak_features}\n"
    else:
        report += "\n[OK] 無原始價格洩漏\n"
    
    report += f"""
---

## [OPT] 策略優化

### 選擇權真實成本參數
- **Delta (ATM)**: 0.5 (價平選擇權)
- **Theta 每日衰減**: 0.3%
- **滑價**: 10 點/次
- **最低交易門檻**: 20 筆

### 最佳參數
- **持有期間**: {safe_get(best_params, 'holding_days'):.0f} 天
- **信心度閾值**: {safe_get(best_params, 'confidence_threshold'):.2f}
- **停損點**: {safe_get(best_params, 'stop_loss'):.1%}
- **停利點**: {safe_get(best_params, 'take_profit'):.1%}

### 訓練集回測績效 (含真實成本)
- **總交易次數**: {safe_get(best_params, 'total_trades'):.0f}
- **勝率**: {safe_get(best_params, 'win_rate'):.2%}
- **平均報酬率**: {safe_get(best_params, 'avg_return'):.2%}
- **Sharpe Ratio**: {safe_get(best_params, 'sharpe_ratio'):.4f}
- **Calmar Ratio**: {safe_get(best_params, 'calmar_ratio'):.4f}
- **最大回撤**: {safe_get(best_params, 'max_drawdown'):.2%}
- **Profit Factor**: {safe_get(best_params, 'profit_factor'):.4f}

---

## [TEST] 測試集驗證

### Out-of-Sample 績效 (含真實成本)
- **總交易次數**: {test_metrics['total_trades']:.0f}
- **勝率**: {test_metrics['win_rate']:.2%}
- **平均報酬率**: {test_metrics['avg_return']:.2%}
- **Sharpe Ratio**: {test_metrics['sharpe_ratio']:.4f}
- **Calmar Ratio**: {test_metrics.get('calmar_ratio', 0):.4f}
- **最大回撤**: {test_metrics['max_drawdown']:.2%}
- **Profit Factor**: {test_metrics['profit_factor']:.4f}

### 訓練集 vs 測試集比較

| 指標 | 訓練集 | 測試集 | 差異 |
|------|--------|--------|------|
| 勝率 | {safe_get(best_params, 'win_rate'):.2%} | {test_metrics['win_rate']:.2%} | {(test_metrics['win_rate']-safe_get(best_params, 'win_rate'))*100:+.2f}% |
| 平均報酬 | {safe_get(best_params, 'avg_return'):.2%} | {test_metrics['avg_return']:.2%} | {(test_metrics['avg_return']-safe_get(best_params, 'avg_return'))*100:+.2f}% |
| Sharpe | {safe_get(best_params, 'sharpe_ratio'):.4f} | {test_metrics['sharpe_ratio']:.4f} | {test_metrics['sharpe_ratio']-safe_get(best_params, 'sharpe_ratio'):+.4f} |
| 評分 | {safe_get(best_params, 'score'):.4f} | {test_metrics['score']:.4f} | {test_metrics['score']-safe_get(best_params, 'score'):+.4f} |

### 過擬合分析
- **評分差異**: {test_metrics.get('overfitting_score', 0):.4f}
"""
    
    if test_metrics.get('overfitting_score', 0) > 0.3:
        report += "- **結論**: !! 嚴重過擬合\n"
    elif test_metrics.get('overfitting_score', 0) > 0.15:
        report += "- **結論**: ! 輕微過擬合\n"
    else:
        report += "- **結論**: OK 泛化良好\n"
    
    report += f"""
---

## [TRADES] 交易分析

### 訓練集交易
- **總交易**: {len(train_trades)} 筆
"""
    if not train_trades.empty:
        report += f"- **Buy Call**: {len(train_trades[train_trades['direction']=='CALL'])} 筆\n"
        report += f"- **Buy Put**: {len(train_trades[train_trades['direction']=='PUT'])} 筆\n"

    report += f"""
### 測試集交易
- **總交易**: {len(test_trades)} 筆
"""
    if not test_trades.empty:
        report += f"- **Buy Call**: {len(test_trades[test_trades['direction']=='CALL'])} 筆\n"
        report += f"- **Buy Put**: {len(test_trades[test_trades['direction']=='PUT'])} 筆\n"

    report += f"""
---

## [GRADE] 策略評級

### 評級標準
- **A+ (>=0.9)**: 優秀, 可考慮實盤
- **A (>=0.8)**: 良好, 謹慎實盤
- **B (>=0.7)**: 中等, 需優化
- **C (>=0.6)**: 不佳, 需大幅改進
- **D (>=0.4)**: 差, 重新設計
- **F (<0.4)**: 失敗, 根本性問題

### 本次評級: **{grade}**

**綜合評分**: {test_metrics['score']:.4f}

"""
    
    if grade in ['A+', 'A']:
        report += """### 建議
[OK] 策略表現優秀:
1. 進行 1-2 個月紙上交易驗證
2. 小額實盤測試
3. 持續監控績效
4. 定期重新訓練模型
"""
    elif grade == 'B':
        report += """### 建議
[!] 策略表現中等:
1. 累積更多歷史資料
2. 優化特徵工程
3. 調整模型參數
4. 延長紙上交易驗證期
"""
    else:
        report += """### 建議
[X] 策略表現不佳:
1. 重新檢視策略邏輯
2. 增加更多特徵
3. 嘗試其他模型
4. 累積更多資料後重新訓練
"""
    
    report += f"""
---

## [!] 風險提示

1. **資料不足**: 目前僅 {data_summary['total_records']} 筆資料, 建議累積至 200+ 筆
2. **市場變化**: 歷史績效不代表未來表現
3. **選擇權風險**: 買方策略可能損失全部權利金
4. **模型限制**: 機器學習模型無法預測極端事件
5. **真實成本**: 報告已包含 Delta/Theta/滑價模擬, 實際成本可能更高

---

## [FILES] 輸出檔案

- `data/train/` - 訓練集資料
- `data/test/` - 測試集資料
- `models/direction_model.pkl` - XGBoost 模型
- `models/lgbm_model.pkl` - LightGBM 模型 (若有)
- `models/meta_learner.pkl` - Stacking Meta Learner (若有)
- `models/feature_cols.json` - 特徵欄位列表
- `results/optimization.json` - 最佳參數
- `results/backtest_train.csv` - 訓練集交易記錄
- `results/backtest_test.csv` - 測試集交易記錄
- `results/training_report.md` - 本報告

---

**報告結束**
"""
    
    # 儲存報告
    report_path = Path("results/training_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"[OK] 訓練報告已生成: {report_path.absolute()}")
    
    return report_path, grade

if __name__ == "__main__":
    try:
        report_path, grade = generate_report()
        
        print("\n" + "="*60)
        print("[SUCCESS] 訓練報告生成完成!")
        print("="*60)
        print(f"\n報告路徑: {report_path}")
        print(f"策略評級: {grade}")
        print("\n[COMPLETE] 自我學習訓練流程全部完成!")
        
    except Exception as e:
        logger.error(f"報告生成失敗: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("[ERROR] 報告生成失敗!")
        print("="*60)

