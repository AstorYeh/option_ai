"""
策略參數優化器
使用網格搜尋或貝葉斯優化來尋找最佳參數組合
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from itertools import product

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
from src.models.ensemble import EnsemblePredictor
from src.backtest.engine import BacktestEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyOptimizer:
    """策略參數優化器"""
    
    def __init__(self):
        """初始化優化器"""
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        logger.info("策略參數優化器初始化完成")
    
    def optimize_holding_period(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        holding_periods: List[int] = None
    ) -> Dict[str, Any]:
        """
        優化持有期間
        
        Args:
            df: 歷史資料
            predictions_df: 預測結果
            holding_periods: 要測試的持有期間列表
        
        Returns:
            優化結果
        """
        if holding_periods is None:
            holding_periods = [1, 3, 5, 7, 10]
        
        logger.info(f"開始優化持有期間,測試範圍: {holding_periods}")
        
        results = []
        engine = BacktestEngine()
        
        for period in holding_periods:
            logger.info(f"測試持有期間: {period} 天")
            
            backtest_results = engine.run_backtest(
                predictions_df,
                df,
                holding_period=period
            )
            
            metrics = backtest_results['metrics']
            
            # 計算綜合評分 (可自訂權重)
            score = (
                metrics['win_rate'] * 0.3 +
                metrics['total_return'] * 0.4 +
                metrics['sharpe_ratio'] * 0.1 +
                (1 - metrics['max_drawdown']) * 0.2
            )
            
            results.append({
                'holding_period': period,
                'score': score,
                'win_rate': metrics['win_rate'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades']
            })
            
            logger.info(f"持有期間 {period} 天 - 評分: {score:.4f}")
        
        # 找出最佳參數
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        logger.info(f"[OK] 最佳持有期間: {best_result['holding_period']} 天")
        logger.info(f"   評分: {best_result['score']:.4f}")
        logger.info(f"   勝率: {best_result['win_rate']:.2%}")
        logger.info(f"   總報酬率: {best_result['total_return']:.2%}")
        
        return {
            'best_holding_period': int(best_result['holding_period']),
            'best_score': float(best_result['score']),
            'all_results': results_df.to_dict('records'),
            'summary': {
                'win_rate': float(best_result['win_rate']),
                'total_return': float(best_result['total_return']),
                'sharpe_ratio': float(best_result['sharpe_ratio']),
                'max_drawdown': float(best_result['max_drawdown'])
            }
        }
    
    def optimize_confidence_threshold(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        holding_period: int = 5,
        thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """
        優化信心度閾值
        
        Args:
            df: 歷史資料
            predictions_df: 預測結果
            holding_period: 持有期間
            thresholds: 要測試的信心度閾值列表
        
        Returns:
            優化結果
        """
        if thresholds is None:
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        logger.info(f"開始優化信心度閾值,測試範圍: {thresholds}")
        
        results = []
        
        for threshold in thresholds:
            # 篩選符合閾值的預測
            filtered_predictions = predictions_df[
                predictions_df['confidence'] >= threshold
            ].copy()
            
            if len(filtered_predictions) < 10:
                logger.warning(f"閾值 {threshold} 的預測數量過少,跳過")
                continue
            
            logger.info(f"測試信心度閾值: {threshold:.2f} (預測數: {len(filtered_predictions)})")
            
            engine = BacktestEngine()
            backtest_results = engine.run_backtest(
                filtered_predictions,
                df,
                holding_period=holding_period
            )
            
            metrics = backtest_results['metrics']
            
            # 計算綜合評分
            score = (
                metrics['win_rate'] * 0.4 +
                metrics['total_return'] * 0.3 +
                metrics['sharpe_ratio'] * 0.15 +
                (1 - metrics['max_drawdown']) * 0.15
            )
            
            results.append({
                'threshold': threshold,
                'score': score,
                'win_rate': metrics['win_rate'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades']
            })
            
            logger.info(f"閾值 {threshold:.2f} - 評分: {score:.4f}")
        
        if not results:
            logger.error("無有效的優化結果")
            return None
        
        # 找出最佳參數
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        logger.info(f"[OK] 最佳信心度閾值: {best_result['threshold']:.2f}")
        logger.info(f"   評分: {best_result['score']:.4f}")
        logger.info(f"   勝率: {best_result['win_rate']:.2%}")
        
        return {
            'best_threshold': float(best_result['threshold']),
            'best_score': float(best_result['score']),
            'all_results': results_df.to_dict('records'),
            'summary': {
                'win_rate': float(best_result['win_rate']),
                'total_return': float(best_result['total_return']),
                'sharpe_ratio': float(best_result['sharpe_ratio']),
                'max_drawdown': float(best_result['max_drawdown']),
                'total_trades': int(best_result['total_trades'])
            }
        }
    
    def grid_search(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]] = None
    ) -> Dict[str, Any]:
        """
        網格搜尋最佳參數組合
        
        Args:
            df: 歷史資料
            predictions_df: 預測結果
            param_grid: 參數網格
        
        Returns:
            優化結果
        """
        if param_grid is None:
            param_grid = {
                'holding_period': [3, 5, 7],
                'confidence_threshold': [0.6, 0.65, 0.7]
            }
        
        logger.info("開始網格搜尋...")
        logger.info(f"參數網格: {param_grid}")
        
        # 生成所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"總共 {total_combinations} 種參數組合")
        
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            param_dict = dict(zip(param_names, params))
            
            logger.info(f"[{i}/{total_combinations}] 測試參數: {param_dict}")
            
            # 篩選預測
            filtered_predictions = predictions_df[
                predictions_df['confidence'] >= param_dict['confidence_threshold']
            ].copy()
            
            if len(filtered_predictions) < 10:
                logger.warning("預測數量過少,跳過")
                continue
            
            # 執行回測
            engine = BacktestEngine()
            backtest_results = engine.run_backtest(
                filtered_predictions,
                df,
                holding_period=param_dict['holding_period']
            )
            
            metrics = backtest_results['metrics']
            
            # 計算評分
            score = (
                metrics['win_rate'] * 0.35 +
                metrics['total_return'] * 0.35 +
                metrics['sharpe_ratio'] * 0.15 +
                (1 - metrics['max_drawdown']) * 0.15
            )
            
            result = {
                **param_dict,
                'score': score,
                'win_rate': metrics['win_rate'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades']
            }
            
            results.append(result)
            
            logger.info(f"   評分: {score:.4f}, 勝率: {metrics['win_rate']:.2%}")
        
        if not results:
            logger.error("無有效的優化結果")
            return None
        
        # 找出最佳參數
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_result = results_df.iloc[best_idx].to_dict()
        
        logger.info("\n" + "="*60)
        logger.info("[OK] 網格搜尋完成!")
        logger.info(f"最佳參數組合:")
        for key in param_names:
            logger.info(f"  {key}: {best_result[key]}")
        logger.info(f"評分: {best_result['score']:.4f}")
        logger.info(f"勝率: {best_result['win_rate']:.2%}")
        logger.info(f"總報酬率: {best_result['total_return']:.2%}")
        logger.info("="*60)
        
        return {
            'best_params': {k: best_result[k] for k in param_names},
            'best_score': float(best_result['score']),
            'all_results': results_df.to_dict('records'),
            'summary': {
                'win_rate': float(best_result['win_rate']),
                'total_return': float(best_result['total_return']),
                'sharpe_ratio': float(best_result['sharpe_ratio']),
                'max_drawdown': float(best_result['max_drawdown']),
                'total_trades': int(best_result['total_trades'])
            }
        }


# 測試程式碼
if __name__ == "__main__":
    from src.models.ensemble import EnsemblePredictor
    
    logger.info("=== 開始策略參數優化 ===")
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料")
        exit(1)
    
    # 生成預測 (簡化版,實際應使用完整預測)
    ensemble = EnsemblePredictor()
    predictions = []
    
    for i in range(100, len(df), 10):
        subset = df.iloc[:i]
        try:
            result = ensemble.predict(subset)
            predictions.append({
                'date': subset.iloc[-1]['date'],
                'direction': result['prediction']['direction'],
                'confidence': result['prediction']['confidence']
            })
        except:
            continue
    
    predictions_df = pd.DataFrame(predictions)
    
    # 執行優化
    optimizer = StrategyOptimizer()
    
    # 1. 優化持有期間
    print("\n" + "="*60)
    print("1. 優化持有期間")
    print("="*60)
    holding_result = optimizer.optimize_holding_period(df, predictions_df)
    
    # 2. 優化信心度閾值
    print("\n" + "="*60)
    print("2. 優化信心度閾值")
    print("="*60)
    threshold_result = optimizer.optimize_confidence_threshold(
        df, 
        predictions_df,
        holding_period=holding_result['best_holding_period']
    )
    
    # 3. 網格搜尋
    print("\n" + "="*60)
    print("3. 網格搜尋最佳參數組合")
    print("="*60)
    grid_result = optimizer.grid_search(df, predictions_df)
    
    print("\n" + "="*60)
    print("[OK] 策略參數優化完成!")
    print("="*60)
