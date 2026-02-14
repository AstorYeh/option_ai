"""
訓練 AI 預測模型
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
from src.features.technical import add_all_technical_indicators
from src.models.direction_model import DirectionPredictor
from src.models.volatility_model import VolatilityPredictor
from src.models.ensemble import EnsemblePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_direction_model():
    """訓練方向性預測模型"""
    logger.info("=== 訓練方向性預測模型 ===")
    
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料,請先執行 daily_update.py")
        return
    
    # 計算技術指標
    df_with_indicators = add_all_technical_indicators(df)
    
    # 訓練模型
    predictor = DirectionPredictor()
    results = predictor.train(df_with_indicators)
    
    print("\n=== 方向性預測模型訓練結果 ===")
    print(f"訓練集準確率: {results['train_accuracy']:.2%}")
    print(f"測試集準確率: {results['test_accuracy']:.2%}")
    print(f"訓練集大小: {results['train_size']} 筆")
    print(f"測試集大小: {results['test_size']} 筆")
    
    print("\n特徵重要性 (Top 10):")
    importance = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in importance[:10]:
        print(f"  {feat}: {imp:.4f}")
    
    # 測試預測
    direction, confidence = predictor.predict(df_with_indicators)
    direction_map = {0: '跌', 1: '盤整', 2: '漲'}
    print(f"\n最新預測:")
    print(f"  方向: {direction_map[direction]}")
    print(f"  信心度: {confidence:.2%}")


def train_volatility_model():
    """訓練波動率預測模型"""
    logger.info("\n=== 訓練波動率預測模型 ===")
    
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料,請先執行 daily_update.py")
        return
    
    # 訓練模型
    predictor = VolatilityPredictor()
    results = predictor.train(df)
    
    print("\n=== 波動率預測模型訓練結果 ===")
    print(f"訓練集 MSE: {results['train_mse']:.6f}")
    print(f"測試集 MSE: {results['test_mse']:.6f}")
    print(f"訓練集 MAE: {results['train_mae']:.6f}")
    print(f"測試集 MAE: {results['test_mae']:.6f}")
    
    # 測試預測
    predicted_vol, current_vol = predictor.predict(df)
    
    if predicted_vol and current_vol:
        print(f"\n最新預測:")
        print(f"  當前波動率: {current_vol:.2%}")
        print(f"  預測波動率: {predicted_vol:.2%}")
        
        if predicted_vol > current_vol * 1.1:
            print("  建議: 波動率預期上升,適合買入選擇權")
        elif predicted_vol < current_vol * 0.9:
            print("  建議: 波動率預期下降,觀望為主")
        else:
            print("  建議: 波動率穩定")


def test_ensemble():
    """測試集成預測系統"""
    logger.info("\n=== 測試集成預測系統 ===")
    
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料,請先執行 daily_update.py")
        return
    
    # 執行集成預測
    ensemble = EnsemblePredictor()
    result = ensemble.predict(df)
    
    print("\n=== 集成預測結果 ===")
    print(f"時間: {result['timestamp']}")
    
    print(f"\n市場資料:")
    print(f"  收盤價: {result['market_data']['close']:.0f}")
    print(f"  漲跌: {result['market_data']['change']:+.0f} ({result['market_data']['change_pct']:+.2f}%)")
    print(f"  RSI: {result['market_data']['rsi']:.1f}")
    print(f"  MACD: {result['market_data']['macd']:.2f}")
    
    print(f"\n方向預測:")
    print(f"  方向: {result['prediction']['direction']}")
    print(f"  信心度: {result['prediction']['confidence']:.1%}")
    print(f"  預期漲跌: {result['prediction']['predicted_change']:+.2f}%")
    
    print(f"\n波動率:")
    print(f"  當前: {result['prediction']['volatility']['current']:.2%}")
    print(f"  預測: {result['prediction']['volatility']['predicted']:.2%}")
    print(f"  趨勢: {result['prediction']['volatility']['trend']}")
    
    print(f"\nLLM 建議:")
    print(f"  動作: {result['llm_advice']['action']}")
    print(f"  理由: {result['llm_advice']['reasoning'][:100]}...")
    print(f"  風險: {result['llm_advice']['risk_level']}")
    
    print(f"\n最終建議:")
    print(f"  動作: {result['final_recommendation']['action']}")
    print(f"  理由: {result['final_recommendation']['reason'][:100]}...")
    print(f"  信心度: {result['final_recommendation']['confidence']:.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練 AI 預測模型")
    parser.add_argument("--direction", action="store_true", help="訓練方向性預測模型")
    parser.add_argument("--volatility", action="store_true", help="訓練波動率預測模型")
    parser.add_argument("--ensemble", action="store_true", help="測試集成預測系統")
    parser.add_argument("--all", action="store_true", help="執行所有訓練與測試")
    
    args = parser.parse_args()
    
    if args.all or (not args.direction and not args.volatility and not args.ensemble):
        train_direction_model()
        train_volatility_model()
        test_ensemble()
    else:
        if args.direction:
            train_direction_model()
        if args.volatility:
            train_volatility_model()
        if args.ensemble:
            test_ensemble()
