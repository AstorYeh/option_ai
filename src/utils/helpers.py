"""
輔助函數
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Union, List
from config.settings import TIMEZONE

def get_taiwan_time() -> datetime:
    """取得台灣時間"""
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)

def is_trading_day(date: Union[str, datetime]) -> bool:
    """
    判斷是否為交易日(排除週末)
    TODO: 需整合台灣期交所休市日曆
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date.weekday() < 5  # 0-4 為週一到週五

def get_last_trading_day(date: Union[str, datetime] = None) -> datetime:
    """取得最近的交易日"""
    if date is None:
        date = get_taiwan_time()
    elif isinstance(date, str):
        date = pd.to_datetime(date)
    
    while not is_trading_day(date):
        date -= timedelta(days=1)
    return date

def calculate_returns(prices: pd.Series) -> pd.Series:
    """計算報酬率"""
    return prices.pct_change()

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """計算對數報酬率"""
    return np.log(prices / prices.shift(1))

def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    資料標準化
    
    Args:
        data: 原始資料
        method: 標準化方法 ('minmax' 或 'zscore')
    """
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def format_currency(amount: float) -> str:
    """格式化金額顯示"""
    return f"NT$ {amount:,.0f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """格式化百分比顯示"""
    return f"{value * 100:.{decimals}f}%"

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    """
    計算 Sharpe Ratio
    
    Args:
        returns: 報酬率序列
        risk_free_rate: 無風險利率(年化)
    """
    excess_returns = returns - risk_free_rate / 252  # 轉換為日報酬
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """計算最大回撤"""
    cumulative = equity_curve.cummax()
    drawdown = (equity_curve - cumulative) / cumulative
    return drawdown.min()

def get_strike_prices(current_price: float, num_strikes: int = 5, tick_size: int = 100) -> List[float]:
    """
    取得建議的履約價列表
    
    Args:
        current_price: 當前台指期價格
        num_strikes: 上下各幾檔
        tick_size: 履約價間距(台指選擇權為 100 點)
    """
    # 找到最接近的履約價
    base_strike = round(current_price / tick_size) * tick_size
    
    strikes = []
    for i in range(-num_strikes, num_strikes + 1):
        strikes.append(base_strike + i * tick_size)
    
    return sorted(strikes)

def calculate_moneyness(spot_price: float, strike_price: float) -> str:
    """
    計算價內外程度
    
    Returns:
        'ITM': In-the-money (價內)
        'ATM': At-the-money (價平)
        'OTM': Out-of-the-money (價外)
    """
    diff = abs(spot_price - strike_price)
    if diff < 50:  # 50 點以內視為價平
        return 'ATM'
    elif spot_price > strike_price:
        return 'ITM'  # 對 Call 而言
    else:
        return 'OTM'

def validate_date_range(start_date: str, end_date: str) -> bool:
    """驗證日期範圍有效性"""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return start < end
    except:
        return False
