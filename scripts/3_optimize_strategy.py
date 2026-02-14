"""
Step 3: 策略參數優化 (含選擇權真實成本模擬)
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
from itertools import product

# 排除原始價格欄位 (與 2_train_models.py 保持一致)
RAW_PRICE_COLS = ['open', 'high', 'low', 'close', 'volume',
                  'spread', 'spread_per', 'atr']
EXCLUDE_COLS = ['date', 'futures_id', 'contract_date', 'trading_session'] + RAW_PRICE_COLS

# 選擇權真實成本參數
OPTION_DELTA = 0.5       # ATM Delta (價平選擇權)
THETA_DAILY = 0.003      # 每天權利金衰減 0.3%
SLIPPAGE_POINTS = 10     # 滑價 10 點
MIN_TRADES = 20          # 最低交易次數門檻


def get_feature_cols(df):
    """取得有效特徵欄位 (排除原始價格和標籤)"""
    # 優先使用儲存的特徵欄位
    feature_cols_path = Path("models/feature_cols.json")
    if feature_cols_path.exists():
        with open(feature_cols_path, 'r', encoding='utf-8') as f:
            saved_cols = json.load(f)
        # 只保留資料中存在的欄位
        return [c for c in saved_cols if c in df.columns]
    
    # 退回到動態偵測
    return [col for col in df.columns 
            if col not in EXCLUDE_COLS
            and not col.startswith('return_') 
            and not col.startswith('label_')]


def backtest_strategy(df, model, params):
    """
    執行回測 (含選擇權真實成本模擬)
    
    模擬選擇權買方策略的真實損益:
    - Delta 槓桿: 選擇權漲跌幅 = 標的漲跌幅 * Delta
    - Theta 衰減: 每天扣除時間價值
    - 滑價: 進出場各承受固定點數滑價
    """
    holding_days = params['holding_days']
    confidence_threshold = params['confidence_threshold']
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    
    # 準備特徵
    feature_cols = get_feature_cols(df)
    
    X = df[feature_cols].fillna(0)
    
    # 預測
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # 轉換預測為 -1, 0, 1
    predictions = predictions - 1
    
    # 計算信心度 (最大機率)
    confidence = probabilities.max(axis=1)
    
    # 模擬交易
    trades = []
    i = 0
    while i < len(df) - holding_days:
        # 檢查信心度
        if confidence[i] < confidence_threshold:
            i += 1
            continue
        
        # 進場
        entry_price = df.iloc[i]['close']
        entry_date = df.iloc[i]['date']
        direction = predictions[i]
        
        if direction == 0:  # 盤整不交易
            i += 1
            continue
        
        # 持有期間
        exit_idx = min(i + holding_days, len(df) - 1)
        exit_idx = int(exit_idx)
        exit_price = df.iloc[exit_idx]['close']
        exit_date = df.iloc[exit_idx]['date']
        
        # 計算標的報酬
        if direction == 1:  # Buy Call
            underlying_pnl = (exit_price - entry_price) / entry_price
        else:  # Buy Put
            underlying_pnl = (entry_price - exit_price) / entry_price
        
        # === 選擇權真實成本模擬 ===
        actual_holding = exit_idx - i
        
        # 1. Delta 槓桿效果
        option_pnl = underlying_pnl * OPTION_DELTA
        
        # 2. Theta 時間衰減
        theta_cost = THETA_DAILY * actual_holding
        option_pnl -= theta_cost
        
        # 3. 滑價成本
        slippage_cost = (SLIPPAGE_POINTS * 2) / entry_price  # 進出場各一次
        option_pnl -= slippage_cost
        
        # 停損停利
        if option_pnl < -stop_loss:
            option_pnl = -stop_loss
        elif option_pnl > take_profit:
            option_pnl = take_profit
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': 'CALL' if direction == 1 else 'PUT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'underlying_pnl': underlying_pnl,
            'pnl_pct': option_pnl,
            'theta_cost': theta_cost,
            'confidence': confidence[i]
        })
        
        i = exit_idx + 1
    
    return pd.DataFrame(trades)

def calculate_metrics(trades_df):
    """計算績效指標 (改進版, 含 Calmar Ratio)"""
    if trades_df.empty or len(trades_df) < 3:
        return {
            'total_trades': len(trades_df) if not trades_df.empty else 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 1.0,
            'profit_factor': 0,
            'calmar_ratio': 0,
            'score': 0
        }
    
    # 基本指標
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 報酬指標
    avg_return = trades_df['pnl_pct'].mean()
    total_return = trades_df['pnl_pct'].sum()
    
    # Sharpe Ratio
    returns_std = trades_df['pnl_pct'].std()
    sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
    
    # 最大回撤
    cumulative_returns = (1 + trades_df['pnl_pct']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 1.0
    max_drawdown = max(max_drawdown, 0.001)  # 避免除以零
    
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = avg_return / max_drawdown if max_drawdown > 0 else 0
    
    # 綜合評分 (Calmar-Based 多因子)
    # 交易次數不足時大幅降權
    trade_penalty = min(total_trades / MIN_TRADES, 1.0)
    
    score = (
        win_rate * 0.25 +
        min(sharpe_ratio / 3, 1) * 0.25 +
        min(calmar_ratio / 5, 1) * 0.20 +
        (1 - min(max_drawdown, 1)) * 0.15 +
        trade_penalty * 0.15
    ) * trade_penalty  # 交易次數不足時整體打折
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'score': score
    }

def optimize_strategy():
    """策略參數優化 (選擇權真實成本版)"""
    logger.info("=== 開始策略優化 (含選擇權真實成本) ===")
    logger.info(f"  Delta: {OPTION_DELTA}, Theta: {THETA_DAILY}/日, 滑價: {SLIPPAGE_POINTS} 點")
    
    # 載入模型
    model = joblib.load("models/direction_model.pkl")
    
    # 載入訓練資料
    train_df = pd.read_csv("data/train/train_labeled.csv")
    logger.info(f"訓練資料: {len(train_df)} 筆")
    
    # 參數網格 (合理化, 符合選擇權真實交易)
    param_grid = {
        'holding_days': [1, 2, 3, 5],
        'confidence_threshold': [0.55, 0.60, 0.65, 0.70],
        'stop_loss': [0.03, 0.05, 0.08, 0.10],
        'take_profit': [0.10, 0.15, 0.20, 0.30]
    }
    
    # 生成所有參數組合
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in product(*param_grid.values())
    ]
    
    logger.info(f"參數組合數: {len(param_combinations)}")
    
    # 測試所有組合
    results = []
    for i, params in enumerate(param_combinations, 1):
        if i % 20 == 0:
            logger.info(f"進度: {i}/{len(param_combinations)}")
        
        trades_df = backtest_strategy(train_df, model, params)
        metrics = calculate_metrics(trades_df)
        
        results.append({
            **params,
            **metrics
        })
    
    # 轉換為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 過濾交易次數不足的組合
    valid_results = results_df[results_df['total_trades'] >= MIN_TRADES]
    
    if valid_results.empty:
        logger.warning(f"[WARN] 沒有足夠交易次數 (>={MIN_TRADES}) 的參數組合, 放寬門檻")
        valid_results = results_df[results_df['total_trades'] >= 5]
    
    if valid_results.empty:
        logger.warning("[WARN] 使用所有結果")
        valid_results = results_df
    
    # 找出最佳參數
    best_idx = valid_results['score'].idxmax()
    best_params = valid_results.iloc[valid_results.index.get_loc(best_idx) if best_idx not in valid_results.index.tolist() else valid_results.index.tolist().index(best_idx)].to_dict()
    best_params = results_df.loc[best_idx].to_dict()
    
    logger.info(f"\n=== 最佳參數 ===")
    logger.info(f"持有期間: {best_params['holding_days']:.0f} 天")
    logger.info(f"信心度閾值: {best_params['confidence_threshold']:.2f}")
    logger.info(f"停損點: {best_params['stop_loss']:.1%}")
    logger.info(f"停利點: {best_params['take_profit']:.1%}")
    
    logger.info(f"\n=== 訓練集績效 (含真實成本) ===")
    logger.info(f"總交易次數: {best_params['total_trades']:.0f}")
    logger.info(f"勝率: {best_params['win_rate']:.2%}")
    logger.info(f"平均報酬率: {best_params['avg_return']:.2%}")
    logger.info(f"Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
    logger.info(f"Calmar Ratio: {best_params['calmar_ratio']:.4f}")
    logger.info(f"最大回撤: {best_params['max_drawdown']:.2%}")
    logger.info(f"綜合評分: {best_params['score']:.4f}")
    
    # 儲存結果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 儲存最佳參數
    with open(results_dir / "optimization.json", 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    # 儲存所有結果
    results_df.to_csv(results_dir / "optimization_results.csv", index=False)
    
    # 執行最佳參數回測並儲存交易記錄
    best_trades = backtest_strategy(train_df, model, best_params)
    best_trades.to_csv(results_dir / "backtest_train.csv", index=False)
    
    logger.info(f"[OK] 優化結果已儲存: {results_dir.absolute()}")
    
    return best_params

if __name__ == "__main__":
    try:
        best_params = optimize_strategy()
        
        print("\n" + "="*60)
        print("[SUCCESS] 策略優化完成! (含選擇權真實成本)")
        print("="*60)
        print(f"\n最佳參數:")
        print(f"  持有期間: {best_params['holding_days']:.0f} 天")
        print(f"  信心度閾值: {best_params['confidence_threshold']:.2f}")
        print(f"  停損: {best_params['stop_loss']:.1%}")
        print(f"  停利: {best_params['take_profit']:.1%}")
        print(f"\n績效 (含 Delta/Theta/滑價):")
        print(f"  勝率: {best_params['win_rate']:.2%}")
        print(f"  平均報酬: {best_params['avg_return']:.2%}")
        print(f"  Calmar Ratio: {best_params['calmar_ratio']:.4f}")
        print("\n下一步: python scripts/4_validate_test.py")
        
    except Exception as e:
        logger.error(f"優化失敗: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("[ERROR] 策略優化失敗!")
        print("="*60)

