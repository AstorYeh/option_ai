"""
技術指標計算模組
包含常用的技術分析指標
"""
import pandas as pd
import numpy as np
from typing import Optional
from config.model_config import FEATURE_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_rsi(df: pd.DataFrame, period: int = None, price_col: str = 'close') -> pd.Series:
    """
    計算 RSI (相對強弱指標)
    
    Args:
        df: 包含價格資料的 DataFrame
        period: 計算週期
        price_col: 價格欄位名稱
    
    Returns:
        RSI 序列
    """
    if period is None:
        period = FEATURE_CONFIG['technical_indicators']['rsi_period']
    
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = None,
    slow: int = None,
    signal: int = None,
    price_col: str = 'close'
) -> tuple:
    """
    計算 MACD (指數平滑異同移動平均線)
    
    Args:
        df: 包含價格資料的 DataFrame
        fast: 快線週期
        slow: 慢線週期
        signal: 訊號線週期
        price_col: 價格欄位名稱
    
    Returns:
        (MACD, Signal, Histogram) 三個序列的 tuple
    """
    config = FEATURE_CONFIG['technical_indicators']
    fast = fast or config['macd_fast']
    slow = slow or config['macd_slow']
    signal = signal or config['macd_signal']
    
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = None,
    std_dev: float = None,
    price_col: str = 'close'
) -> tuple:
    """
    計算布林通道
    
    Args:
        df: 包含價格資料的 DataFrame
        period: 計算週期
        std_dev: 標準差倍數
        price_col: 價格欄位名稱
    
    Returns:
        (Upper Band, Middle Band, Lower Band) 三個序列的 tuple
    """
    config = FEATURE_CONFIG['technical_indicators']
    period = period or config['bb_period']
    std_dev = std_dev or config['bb_std']
    
    middle_band = df[price_col].rolling(window=period).mean()
    std = df[price_col].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    計算 ATR (真實波動幅度)
    
    Args:
        df: 包含 OHLC 資料的 DataFrame
        period: 計算週期
    
    Returns:
        ATR 序列
    """
    if period is None:
        period = FEATURE_CONFIG['technical_indicators']['atr_period']
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_moving_averages(
    df: pd.DataFrame,
    periods: list = None,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    計算多個週期的移動平均線偏離度 (相對比率, 避免原始價格洩漏)
    
    Args:
        df: 包含價格資料的 DataFrame
        periods: 週期列表
        price_col: 價格欄位名稱
    
    Returns:
        包含各週期 MA 偏離度的 DataFrame
    """
    if periods is None:
        periods = FEATURE_CONFIG['technical_indicators']['ma_periods']
    
    result = pd.DataFrame(index=df.index)
    
    for period in periods:
        ma = df[price_col].rolling(window=period).mean()
        # 使用偏離比率而非原始 MA 值
        result[f'ma{period}_dev'] = (df[price_col] - ma) / ma
    
    return result


def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算成交量相關指標
    
    Args:
        df: 包含成交量資料的 DataFrame
    
    Returns:
        包含成交量指標的 DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    # 成交量移動平均
    volume_ma5 = df['volume'].rolling(window=5).mean()
    volume_ma20 = df['volume'].rolling(window=20).mean()
    
    # 成交量比率 (標準化)
    result['volume_ratio_5'] = df['volume'] / volume_ma5
    result['volume_ratio_20'] = df['volume'] / volume_ma20
    
    # 成交量變化率
    result['volume_change'] = df['volume'].pct_change()
    result['volume_change_5'] = df['volume'].pct_change(5)
    
    # OBV 標準化 (使用變化率而非絕對值)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ma = obv.rolling(window=20).mean()
    result['obv_dev'] = (obv - obv_ma) / obv_ma.abs().clip(lower=1)
    
    return result


def calculate_momentum_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    計算動量指標 (全部使用百分比, 避免絕對值洩漏)
    
    Args:
        df: 包含價格資料的 DataFrame
        price_col: 價格欄位名稱
    
    Returns:
        包含動量指標的 DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    # ROC (變動率) - 已經是百分比
    result['roc_5'] = df[price_col].pct_change(5) * 100
    result['roc_10'] = df[price_col].pct_change(10) * 100
    result['roc_20'] = df[price_col].pct_change(20) * 100
    
    # 威廉指標 (Williams %R) - 已經是百分比
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    result['williams_r'] = -100 * (high_14 - df[price_col]) / (high_14 - low_14)
    
    return result


def calculate_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算價格衍生特徵 (替代原始 OHLC 價格)
    
    所有特徵均為比率或百分比形式, 不含絕對價格,
    避免模型「記住」特定價格區間造成過擬合。
    
    Args:
        df: 包含 OHLCV 資料的 DataFrame
    
    Returns:
        包含價格衍生特徵的 DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    # === K線形態特徵 ===
    # 日報酬率
    result['daily_return'] = df['close'].pct_change()
    
    # 盤中波幅 (high-low)/close
    result['intraday_range'] = (df['high'] - df['low']) / df['close']
    
    # 跳空幅度 (open - prev_close) / prev_close
    result['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 上影線比率
    body_top = df[['open', 'close']].max(axis=1)
    body_bottom = df[['open', 'close']].min(axis=1)
    total_range = (df['high'] - df['low']).clip(lower=0.01)
    result['upper_shadow'] = (df['high'] - body_top) / total_range
    
    # 下影線比率
    result['lower_shadow'] = (body_bottom - df['low']) / total_range
    
    # 實體比率 (正=收紅, 負=收黑)
    result['body_ratio'] = (df['close'] - df['open']) / total_range
    
    # === 相對位置特徵 ===
    # 價格位置 (20日高低點之間的位置)
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    range_20 = (high_20 - low_20).clip(lower=0.01)
    result['price_position'] = (df['close'] - low_20) / range_20
    
    # ATR 標準化 (ATR / close)
    atr = calculate_atr(df)
    result['atr_pct'] = atr / df['close']
    
    # 布林通道位置 (%B 指標)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
    bb_range = (bb_upper - bb_lower).clip(lower=0.01)
    result['bb_position'] = (df['close'] - bb_lower) / bb_range
    result['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # === 波動率特徵 ===
    # 歷史波動率 (年化)
    result['historical_volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 波動率變化 (當前 vs 過去)
    hv_5 = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
    hv_20 = result['historical_volatility']
    result['volatility_ratio'] = hv_5 / hv_20.clip(lower=0.001)
    
    logger.info(f"[OK] 價格衍生特徵計算完成,共 {len(result.columns)} 個特徵")
    
    return result


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    一次性加入所有技術指標
    
    Args:
        df: 原始 OHLCV 資料
    
    Returns:
        加入所有技術指標的 DataFrame
    """
    logger.info("計算技術指標...")
    
    result = df.copy()
    
    # RSI
    result['rsi'] = calculate_rsi(df)
    
    # MACD (標準化: 除以收盤價)
    macd, signal, histogram = calculate_macd(df)
    result['macd_pct'] = macd / df['close']
    result['macd_signal_pct'] = signal / df['close']
    result['macd_histogram_pct'] = histogram / df['close']
    
    # ATR (保留原始值供其他模組使用)
    result['atr'] = calculate_atr(df)
    
    # 移動平均線偏離度 (相對比率)
    ma_df = calculate_moving_averages(df)
    result = pd.concat([result, ma_df], axis=1)
    
    # 成交量指標 (標準化)
    volume_df = calculate_volume_indicators(df)
    result = pd.concat([result, volume_df], axis=1)
    
    # 動量指標 (百分比)
    momentum_df = calculate_momentum_indicators(df)
    result = pd.concat([result, momentum_df], axis=1)
    
    # 價格衍生特徵 (替代原始 OHLC)
    price_df = calculate_price_derived_features(df)
    result = pd.concat([result, price_df], axis=1)
    
    logger.info(f"[OK] 技術指標計算完成,共 {len(result.columns) - len(df.columns)} 個指標")
    
    return result


# 測試程式碼
if __name__ == "__main__":
    # 建立測試資料
    dates = pd.date_range('2024-01-01', periods=100)
    test_df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 18000,
        'high': np.random.randn(100).cumsum() + 18100,
        'low': np.random.randn(100).cumsum() + 17900,
        'close': np.random.randn(100).cumsum() + 18000,
        'volume': np.random.randint(50000, 150000, 100)
    })
    
    # 計算所有技術指標
    result = add_all_technical_indicators(test_df)
    print(result.tail())
    print(f"\n總共 {len(result.columns)} 個欄位")
