"""
回測引擎
用於驗證選擇權買方策略的歷史績效
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.utils.logger import get_logger
from config.model_config import BACKTEST_CONFIG

logger = get_logger(__name__)


@dataclass
class Trade:
    """交易記錄"""
    entry_date: str
    exit_date: str
    direction: str  # 'call' or 'put'
    strike_price: float
    entry_price: float  # 權利金
    exit_price: float
    quantity: int
    profit_loss: float
    return_pct: float
    holding_days: int


class BacktestEngine:
    """回測引擎"""
    
    def __init__(self):
        """初始化回測引擎"""
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.initial_capital = 100000  # 初始資金 10 萬
        self.current_capital = self.initial_capital
        
        logger.info("回測引擎初始化完成")
    
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        actual_data: pd.DataFrame,
        holding_period: int = 5
    ) -> Dict[str, Any]:
        """
        執行回測
        
        Args:
            predictions: 預測結果 DataFrame (包含 date, direction, confidence)
            actual_data: 實際市場資料 DataFrame
            holding_period: 持有天數
        
        Returns:
            回測結果字典
        """
        logger.info(f"開始回測,預測筆數: {len(predictions)}, 持有期間: {holding_period} 天")
        
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_capital = self.initial_capital
        
        # 合併預測與實際資料
        df = predictions.merge(actual_data, on='date', how='inner')
        
        for idx, row in df.iterrows():
            # 檢查是否有進場訊號
            if row['confidence'] < 0.6:  # 信心度過低,跳過
                continue
            
            # 計算進場與出場
            entry_date = row['date']
            exit_date = self._get_exit_date(actual_data, entry_date, holding_period)
            
            if exit_date is None:
                continue
            
            # 模擬交易
            trade = self._simulate_trade(
                entry_date=entry_date,
                exit_date=exit_date,
                direction=row['direction'],
                entry_price_ref=row['close'],
                actual_data=actual_data,
                holding_period=holding_period
            )
            
            if trade:
                self.trades.append(trade)
                self.current_capital += trade.profit_loss
                self.equity_curve.append(self.current_capital)
        
        # 計算績效指標
        metrics = self._calculate_metrics()
        
        logger.info(f"回測完成,總交易次數: {len(self.trades)}")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital
        }
    
    def _get_exit_date(
        self,
        data: pd.DataFrame,
        entry_date: str,
        holding_period: int
    ) -> str:
        """取得出場日期"""
        try:
            entry_idx = data[data['date'] == entry_date].index[0]
            exit_idx = entry_idx + holding_period
            
            if exit_idx >= len(data):
                return None
            
            return data.iloc[exit_idx]['date']
        except:
            return None
    
    def _simulate_trade(
        self,
        entry_date: str,
        exit_date: str,
        direction: str,
        entry_price_ref: float,
        actual_data: pd.DataFrame,
        holding_period: int
    ) -> Trade:
        """
        模擬交易
        
        Args:
            entry_date: 進場日期
            exit_date: 出場日期
            direction: 方向 ('bullish' -> call, 'bearish' -> put)
            entry_price_ref: 進場參考價格
            actual_data: 實際資料
            holding_period: 持有天數
        
        Returns:
            Trade 物件
        """
        # 取得進場與出場價格
        entry_row = actual_data[actual_data['date'] == entry_date].iloc[0]
        exit_row = actual_data[actual_data['date'] == exit_date].iloc[0]
        
        # 簡化模擬: 使用價格變動來估算權利金變化
        price_change = exit_row['close'] - entry_row['close']
        price_change_pct = price_change / entry_row['close']
        
        # 選擇權方向
        option_type = 'call' if direction == 'bullish' else 'put'
        
        # 模擬權利金 (簡化: 假設為標的價格的 2%)
        entry_premium = entry_row['close'] * 0.02
        
        # 計算出場權利金 (簡化模型)
        if option_type == 'call':
            # Call: 價格上漲有利
            premium_change_pct = price_change_pct * 5  # 槓桿效果
        else:
            # Put: 價格下跌有利
            premium_change_pct = -price_change_pct * 5
        
        exit_premium = entry_premium * (1 + premium_change_pct)
        exit_premium = max(0, exit_premium)  # 權利金不能為負
        
        # 計算損益 (每口 50 點,1 口)
        quantity = 1
        profit_loss = (exit_premium - entry_premium) * 50 * quantity
        return_pct = (exit_premium - entry_premium) / entry_premium if entry_premium > 0 else 0
        
        return Trade(
            entry_date=str(entry_date),
            exit_date=str(exit_date),
            direction=option_type,
            strike_price=entry_row['close'],  # 簡化: 使用平值
            entry_price=entry_premium,
            exit_price=exit_premium,
            quantity=quantity,
            profit_loss=profit_loss,
            return_pct=return_pct,
            holding_days=holding_period
        )
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """計算績效指標"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # 基本統計
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # 報酬率
        returns = [t.return_pct for t in self.trades]
        avg_return = np.mean(returns) if returns else 0
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe Ratio (簡化: 假設無風險利率為 0)
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # 年化
        
        # Profit Factor
        total_profit = sum([t.profit_loss for t in winning_trades])
        total_loss = abs(sum([t.profit_loss for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_profit': total_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss': total_loss / len(losing_trades) if losing_trades else 0
        }
    
    def _calculate_max_drawdown(self) -> float:
        """計算最大回撤"""
        if len(self.equity_curve) < 2:
            return 0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成回測報告
        
        Args:
            results: 回測結果
        
        Returns:
            報告文字
        """
        metrics = results['metrics']
        
        report = f"""
╔══════════════════════════════════════════╗
║          回測績效報告                    ║
╚══════════════════════════════════════════╝

【資金狀況】
初始資金: ${results['initial_capital']:,.0f}
最終資金: ${results['final_capital']:,.0f}
總報酬率: {metrics['total_return']:.2%}

【交易統計】
總交易次數: {metrics['total_trades']}
獲利次數: {metrics['winning_trades']}
虧損次數: {metrics['losing_trades']}
勝率: {metrics['win_rate']:.2%}

【報酬分析】
平均報酬率: {metrics['avg_return']:.2%}
平均獲利: ${metrics['avg_profit']:,.0f}
平均虧損: ${metrics['avg_loss']:,.0f}
獲利因子: {metrics['profit_factor']:.2f}

【風險指標】
最大回撤: {metrics['max_drawdown']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

【評級】
{self._get_rating(metrics)}
"""
        return report
    
    def _get_rating(self, metrics: Dict[str, float]) -> str:
        """評級系統"""
        score = 0
        
        # 勝率評分
        if metrics['win_rate'] > 0.6:
            score += 3
        elif metrics['win_rate'] > 0.5:
            score += 2
        elif metrics['win_rate'] > 0.4:
            score += 1
        
        # 總報酬評分
        if metrics['total_return'] > 0.2:
            score += 3
        elif metrics['total_return'] > 0.1:
            score += 2
        elif metrics['total_return'] > 0:
            score += 1
        
        # Sharpe Ratio 評分
        if metrics['sharpe_ratio'] > 1.5:
            score += 2
        elif metrics['sharpe_ratio'] > 1.0:
            score += 1
        
        # 評級
        if score >= 7:
            return "⭐⭐⭐⭐⭐ 優秀"
        elif score >= 5:
            return "⭐⭐⭐⭐ 良好"
        elif score >= 3:
            return "⭐⭐⭐ 普通"
        elif score >= 1:
            return "⭐⭐ 待改進"
        else:
            return "⭐ 不建議"


# 測試程式碼
if __name__ == "__main__":
    from src.data.database import Database
    from src.models.ensemble import EnsemblePredictor
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if not df.empty:
        # 生成預測
        ensemble = EnsemblePredictor()
        
        # 模擬歷史預測
        predictions = []
        for i in range(50, len(df), 5):  # 每 5 天預測一次
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
        
        # 執行回測
        engine = BacktestEngine()
        results = engine.run_backtest(predictions_df, df, holding_period=5)
        
        # 顯示報告
        print(engine.generate_report(results))
        
        # 顯示部分交易記錄
        print("\n【最近 5 筆交易】")
        for trade in results['trades'][-5:]:
            print(f"{trade.entry_date} -> {trade.exit_date}: "
                  f"{trade.direction.upper()} "
                  f"損益: ${trade.profit_loss:+,.0f} ({trade.return_pct:+.1%})")
