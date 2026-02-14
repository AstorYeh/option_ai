"""
資金管理模組
實作 Kelly Criterion 與動態部位管理
"""
import numpy as np
from typing import Dict, Optional
from loguru import logger


class MoneyManager:
    """資金管理器"""
    
    def __init__(
        self,
        total_capital: float,
        max_position_pct: float = 0.05,  # 單次最大投入 5%
        max_positions: int = 3,  # 最多同時持有 3 口
        kelly_fraction: float = 0.25  # Kelly 公式的保守係數
    ):
        """
        初始化資金管理器
        
        Args:
            total_capital: 總資金
            max_position_pct: 單次最大投入比例
            max_positions: 最多同時持有部位數
            kelly_fraction: Kelly 公式係數 (0.25 = 1/4 Kelly)
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.kelly_fraction = kelly_fraction
        
        self.current_positions = 0
        self.available_capital = total_capital
        
        logger.info(f"資金管理器初始化: 總資金 {total_capital:,.0f}, "
                   f"單次最大 {max_position_pct:.1%}, "
                   f"最多 {max_positions} 口")
    
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        計算 Kelly Criterion 建議部位大小
        
        Args:
            win_rate: 勝率
            avg_win: 平均獲利率
            avg_loss: 平均虧損率
        
        Returns:
            建議投入比例 (0-1)
        """
        if avg_loss == 0:
            return 0.0
        
        # Kelly Formula: f = (p*b - q) / b
        # p = 勝率, q = 敗率, b = 賠率 (avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (p * b - q) / b
        
        # 保守調整
        kelly = max(0, kelly) * self.kelly_fraction
        
        # 不超過最大限制
        kelly = min(kelly, self.max_position_pct)
        
        logger.info(f"Kelly 計算: 勝率={win_rate:.2%}, "
                   f"賠率={b:.2f}, "
                   f"建議部位={kelly:.2%}")
        
        return kelly
    
    def get_position_size(
        self,
        confidence: float,
        win_rate: float = 0.88,
        avg_win: float = 0.6892,
        avg_loss: float = 0.10
    ) -> Dict[str, float]:
        """
        根據信心度與歷史績效計算建議部位
        
        Args:
            confidence: 模型信心度 (0-1)
            win_rate: 歷史勝率
            avg_win: 歷史平均獲利率
            avg_loss: 歷史平均虧損率
        
        Returns:
            {
                'position_pct': 建議投入比例,
                'position_amount': 建議投入金額,
                'max_loss': 最大可能虧損,
                'expected_return': 期望報酬
            }
        """
        # 檢查是否還能開倉
        if self.current_positions >= self.max_positions:
            logger.warning(f"已達最大部位數 {self.max_positions}")
            return {
                'position_pct': 0.0,
                'position_amount': 0.0,
                'max_loss': 0.0,
                'expected_return': 0.0,
                'reason': '已達最大部位數'
            }
        
        # 計算 Kelly 建議部位
        kelly_size = self.calculate_kelly_size(win_rate, avg_win, avg_loss)
        
        # 根據信心度調整
        # 信心度 0.5 -> 100% Kelly
        # 信心度 0.7 -> 120% Kelly
        # 信心度 0.9 -> 140% Kelly
        confidence_multiplier = 0.5 + (confidence - 0.5) * 2
        adjusted_size = kelly_size * confidence_multiplier
        
        # 確保不超過限制
        adjusted_size = min(adjusted_size, self.max_position_pct)
        
        # 計算實際金額
        position_amount = self.available_capital * adjusted_size
        
        # 計算風險與報酬
        max_loss = position_amount * avg_loss
        expected_return = position_amount * (win_rate * avg_win - (1 - win_rate) * avg_loss)
        
        result = {
            'position_pct': adjusted_size,
            'position_amount': position_amount,
            'max_loss': max_loss,
            'expected_return': expected_return,
            'confidence': confidence,
            'kelly_size': kelly_size
        }
        
        logger.info(f"建議部位: {adjusted_size:.2%} "
                   f"({position_amount:,.0f} 元), "
                   f"最大虧損 {max_loss:,.0f}, "
                   f"期望報酬 {expected_return:,.0f}")
        
        return result
    
    def open_position(self, amount: float) -> bool:
        """
        開倉
        
        Args:
            amount: 投入金額
        
        Returns:
            是否成功開倉
        """
        if amount > self.available_capital:
            logger.error(f"資金不足: 需要 {amount:,.0f}, 可用 {self.available_capital:,.0f}")
            return False
        
        if self.current_positions >= self.max_positions:
            logger.error(f"已達最大部位數 {self.max_positions}")
            return False
        
        self.available_capital -= amount
        self.current_positions += 1
        
        logger.info(f"開倉成功: 投入 {amount:,.0f}, "
                   f"剩餘 {self.available_capital:,.0f}, "
                   f"部位數 {self.current_positions}")
        
        return True
    
    def close_position(self, pnl: float) -> None:
        """
        平倉
        
        Args:
            pnl: 損益 (正數為獲利,負數為虧損)
        """
        self.available_capital += pnl
        self.total_capital += pnl
        self.current_positions -= 1
        
        logger.info(f"平倉: 損益 {pnl:+,.0f}, "
                   f"總資金 {self.total_capital:,.0f}, "
                   f"部位數 {self.current_positions}")
    
    def get_status(self) -> Dict[str, float]:
        """取得當前狀態"""
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'current_positions': self.current_positions,
            'capital_usage': 1 - self.available_capital / self.total_capital,
            'max_positions': self.max_positions
        }


# 測試程式碼
if __name__ == "__main__":
    # 初始化資金管理器
    mm = MoneyManager(total_capital=100000)
    
    # 測試不同信心度的建議部位
    for confidence in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\n信心度 {confidence:.0%}:")
        result = mm.get_position_size(confidence)
        print(f"  建議投入: {result['position_amount']:,.0f} 元 ({result['position_pct']:.2%})")
        print(f"  最大虧損: {result['max_loss']:,.0f} 元")
        print(f"  期望報酬: {result['expected_return']:,.0f} 元")
