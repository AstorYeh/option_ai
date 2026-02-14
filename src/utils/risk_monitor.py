"""
風險監控模組
實作波動率異常偵測與緊急停損機制
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger


class RiskMonitor:
    """風險監控器"""
    
    def __init__(
        self,
        atr_threshold: float = 2.0,  # ATR 異常閾值 (2倍歷史平均)
        intraday_loss_threshold: float = 0.03,  # 盤中最大虧損 3%
        max_consecutive_losses: int = 3,  # 最大連續虧損次數
        volatility_window: int = 20  # 波動率計算窗口
    ):
        """
        初始化風險監控器
        
        Args:
            atr_threshold: ATR 異常倍數
            intraday_loss_threshold: 盤中最大虧損比例
            max_consecutive_losses: 最大連續虧損次數
            volatility_window: 波動率計算窗口
        """
        self.atr_threshold = atr_threshold
        self.intraday_loss_threshold = intraday_loss_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self.volatility_window = volatility_window
        
        self.consecutive_losses = 0
        self.trade_history: List[Dict] = []
        
        logger.info(f"風險監控器初始化: ATR閾值={atr_threshold}x, "
                   f"盤中停損={intraday_loss_threshold:.1%}, "
                   f"最大連虧={max_consecutive_losses}次")
    
    def check_volatility_anomaly(
        self,
        df: pd.DataFrame,
        current_atr: Optional[float] = None
    ) -> Dict[str, any]:
        """
        檢查波動率異常
        
        Args:
            df: 包含 ATR 的歷史資料
            current_atr: 當前 ATR (若為 None 則使用最新值)
        
        Returns:
            {
                'is_anomaly': 是否異常,
                'current_atr': 當前 ATR,
                'avg_atr': 平均 ATR,
                'ratio': ATR 比率,
                'action': 建議動作
            }
        """
        if 'atr' not in df.columns:
            logger.warning("資料中無 ATR 欄位")
            return {'is_anomaly': False, 'action': 'CONTINUE'}
        
        # 計算歷史平均 ATR
        avg_atr = df['atr'].tail(self.volatility_window).mean()
        
        # 取得當前 ATR
        if current_atr is None:
            current_atr = df['atr'].iloc[-1]
        
        # 計算比率
        ratio = current_atr / avg_atr if avg_atr > 0 else 0
        
        # 判斷是否異常
        is_anomaly = ratio > self.atr_threshold
        
        result = {
            'is_anomaly': is_anomaly,
            'current_atr': current_atr,
            'avg_atr': avg_atr,
            'ratio': ratio,
            'action': 'STOP' if is_anomaly else 'CONTINUE'
        }
        
        if is_anomaly:
            logger.warning(f"[ALERT] 波動率異常! "
                          f"當前 ATR={current_atr:.2f}, "
                          f"平均 ATR={avg_atr:.2f}, "
                          f"比率={ratio:.2f}x")
        else:
            logger.info(f"波動率正常: ATR比率={ratio:.2f}x")
        
        return result
    
    def check_intraday_loss(
        self,
        entry_price: float,
        current_price: float,
        position_type: str  # 'CALL' or 'PUT'
    ) -> Dict[str, any]:
        """
        檢查盤中虧損
        
        Args:
            entry_price: 進場價格
            current_price: 當前價格
            position_type: 部位類型
        
        Returns:
            {
                'should_stop': 是否應該停損,
                'loss_pct': 虧損比例,
                'action': 建議動作
            }
        """
        # 計算虧損比例
        if position_type == 'CALL':
            loss_pct = (current_price - entry_price) / entry_price
        else:  # PUT
            loss_pct = (entry_price - current_price) / entry_price
        
        # 判斷是否觸發停損
        should_stop = loss_pct < -self.intraday_loss_threshold
        
        result = {
            'should_stop': should_stop,
            'loss_pct': loss_pct,
            'action': 'EMERGENCY_STOP' if should_stop else 'HOLD'
        }
        
        if should_stop:
            logger.error(f"[EMERGENCY] 盤中虧損觸發停損! "
                        f"虧損 {loss_pct:.2%} > {self.intraday_loss_threshold:.2%}")
        
        return result
    
    def record_trade(
        self,
        entry_date: datetime,
        exit_date: datetime,
        pnl: float,
        return_pct: float
    ) -> None:
        """
        記錄交易結果
        
        Args:
            entry_date: 進場日期
            exit_date: 出場日期
            pnl: 損益
            return_pct: 報酬率
        """
        is_win = pnl > 0
        
        # 更新連續虧損計數
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # 記錄交易
        trade = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'pnl': pnl,
            'return_pct': return_pct,
            'is_win': is_win,
            'consecutive_losses': self.consecutive_losses
        }
        
        self.trade_history.append(trade)
        
        logger.info(f"記錄交易: {'獲利' if is_win else '虧損'} {return_pct:+.2%}, "
                   f"連續虧損 {self.consecutive_losses} 次")
    
    def check_consecutive_losses(self) -> Dict[str, any]:
        """
        檢查連續虧損
        
        Returns:
            {
                'should_pause': 是否應該暫停交易,
                'consecutive_losses': 連續虧損次數,
                'action': 建議動作
            }
        """
        should_pause = self.consecutive_losses >= self.max_consecutive_losses
        
        result = {
            'should_pause': should_pause,
            'consecutive_losses': self.consecutive_losses,
            'action': 'PAUSE_TRADING' if should_pause else 'CONTINUE'
        }
        
        if should_pause:
            logger.error(f"[ALERT] 連續虧損 {self.consecutive_losses} 次! "
                        f"建議暫停交易,重新檢視策略")
        
        return result
    
    def get_risk_summary(self) -> Dict[str, any]:
        """
        取得風險摘要
        
        Returns:
            風險指標摘要
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'max_consecutive_losses': 0,
                'current_consecutive_losses': self.consecutive_losses
            }
        
        df = pd.DataFrame(self.trade_history)
        
        return {
            'total_trades': len(df),
            'win_rate': df['is_win'].mean(),
            'avg_return': df['return_pct'].mean(),
            'max_return': df['return_pct'].max(),
            'min_return': df['return_pct'].min(),
            'max_consecutive_losses': df['consecutive_losses'].max(),
            'current_consecutive_losses': self.consecutive_losses
        }


# 測試程式碼
if __name__ == "__main__":
    # 建立測試資料
    dates = pd.date_range('2024-01-01', periods=100)
    test_df = pd.DataFrame({
        'date': dates,
        'close': np.random.randn(100).cumsum() + 18000,
        'atr': np.random.randn(100) * 50 + 200
    })
    
    # 初始化風險監控器
    rm = RiskMonitor()
    
    # 測試波動率檢查
    print("\n=== 波動率檢查 ===")
    result = rm.check_volatility_anomaly(test_df)
    print(f"是否異常: {result['is_anomaly']}")
    print(f"ATR 比率: {result['ratio']:.2f}x")
    
    # 測試盤中虧損檢查
    print("\n=== 盤中虧損檢查 ===")
    result = rm.check_intraday_loss(entry_price=100, current_price=95, position_type='CALL')
    print(f"應該停損: {result['should_stop']}")
    print(f"虧損比例: {result['loss_pct']:.2%}")
    
    # 測試交易記錄
    print("\n=== 交易記錄 ===")
    for i in range(5):
        pnl = np.random.randn() * 1000
        rm.record_trade(
            entry_date=datetime.now(),
            exit_date=datetime.now() + timedelta(days=1),
            pnl=pnl,
            return_pct=pnl / 10000
        )
    
    summary = rm.get_risk_summary()
    print(f"總交易: {summary['total_trades']}")
    print(f"勝率: {summary['win_rate']:.2%}")
    print(f"平均報酬: {summary['avg_return']:.2%}")
