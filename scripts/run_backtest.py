"""
執行回測測試
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
from src.models.ensemble import EnsemblePredictor
from src.backtest.engine import BacktestEngine
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def run_backtest_test():
    """執行回測測試"""
    logger.info("=== 開始回測測試 ===")
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料,請先執行 daily_update.py")
        return
    
    logger.info(f"載入 {len(df)} 筆歷史資料")
    
    # 生成預測
    ensemble = EnsemblePredictor()
    
    # 模擬歷史預測 (每 5 天預測一次)
    predictions = []
    logger.info("生成歷史預測...")
    
    for i in range(100, len(df), 5):  # 從第 100 筆開始,每 5 天預測一次
        subset = df.iloc[:i]
        try:
            result = ensemble.predict(subset)
            predictions.append({
                'date': subset.iloc[-1]['date'],
                'direction': result['prediction']['direction'],
                'confidence': result['prediction']['confidence']
            })
            
            if len(predictions) % 10 == 0:
                logger.info(f"已生成 {len(predictions)} 筆預測...")
        except Exception as e:
            logger.warning(f"預測失敗: {e}")
            continue
    
    predictions_df = pd.DataFrame(predictions)
    logger.info(f"共生成 {len(predictions_df)} 筆預測")
    
    # 執行回測
    logger.info("執行回測...")
    engine = BacktestEngine()
    results = engine.run_backtest(predictions_df, df, holding_period=5)
    
    # 顯示報告
    print(engine.generate_report(results))
    
    # 顯示部分交易記錄
    print("\n【最近 10 筆交易】")
    for trade in results['trades'][-10:]:
        print(f"{trade.entry_date} -> {trade.exit_date}: "
              f"{trade.direction.upper():4s} "
              f"損益: ${trade.profit_loss:+8,.0f} ({trade.return_pct:+6.1%})")
    
    # 儲存結果
    logger.info(f"回測完成,總交易次數: {len(results['trades'])}")
    
    return results


if __name__ == "__main__":
    results = run_backtest_test()
