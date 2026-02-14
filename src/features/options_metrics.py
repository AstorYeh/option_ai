"""
選擇權專屬指標計算模組
包含隱含波動率、歷史波動率、Put/Call Ratio 等
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy.stats import norm
from config.model_config import FEATURE_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_historical_volatility(
    df: pd.DataFrame,
    window: int = None,
    price_col: str = 'close',
    annualize: bool = True
) -> pd.Series:
    """
    計算歷史波動率 (Historical Volatility)
    
    Args:
        df: 包含價格資料的 DataFrame
        window: 計算窗口(交易日)
        price_col: 價格欄位名稱
        annualize: 是否年化
    
    Returns:
        歷史波動率序列
    """
    if window is None:
        window = FEATURE_CONFIG['volatility_metrics']['hv_window']
    
    # 計算對數報酬率
    log_returns = np.log(df[price_col] / df[price_col].shift(1))
    
    # 計算標準差
    hv = log_returns.rolling(window=window).std()
    
    # 年化(假設一年 252 個交易日)
    if annualize:
        hv = hv * np.sqrt(252)
    
    return hv


def calculate_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    計算 Parkinson 波動率(使用高低價資訊,更準確)
    
    Args:
        df: 包含 OHLC 資料的 DataFrame
        window: 計算窗口
    
    Returns:
        Parkinson 波動率序列
    """
    hl_ratio = np.log(df['high'] / df['low']) ** 2
    parkinson_vol = np.sqrt(hl_ratio.rolling(window=window).mean() / (4 * np.log(2)))
    
    # 年化
    parkinson_vol = parkinson_vol * np.sqrt(252)
    
    return parkinson_vol


def calculate_put_call_ratio(options_df: pd.DataFrame) -> float:
    """
    計算 Put/Call Ratio (市場情緒指標)
    
    Args:
        options_df: 選擇權資料 DataFrame (需包含 contract_type 和 volume)
    
    Returns:
        Put/Call Ratio
    """
    if options_df.empty:
        return np.nan
    
    put_volume = options_df[options_df['contract_type'] == 'put']['volume'].sum()
    call_volume = options_df[options_df['contract_type'] == 'call']['volume'].sum()
    
    if call_volume == 0:
        return np.nan
    
    pcr = put_volume / call_volume
    
    logger.debug(f"Put/Call Ratio: {pcr:.2f} (Put: {put_volume}, Call: {call_volume})")
    
    return pcr


def calculate_put_call_oi_ratio(options_df: pd.DataFrame) -> float:
    """
    計算未平倉量 Put/Call Ratio
    
    Args:
        options_df: 選擇權資料 DataFrame
    
    Returns:
        未平倉量 Put/Call Ratio
    """
    if options_df.empty:
        return np.nan
    
    put_oi = options_df[options_df['contract_type'] == 'put']['open_interest'].sum()
    call_oi = options_df[options_df['contract_type'] == 'call']['open_interest'].sum()
    
    if call_oi == 0:
        return np.nan
    
    return put_oi / call_oi


def calculate_max_pain(options_df: pd.DataFrame, spot_price: float) -> float:
    """
    計算最大痛點 (Max Pain)
    選擇權到期時,造成選擇權買方最大損失的價位
    
    Args:
        options_df: 選擇權資料 DataFrame
        spot_price: 當前現貨價格
    
    Returns:
        最大痛點價位
    """
    if options_df.empty:
        return spot_price
    
    # 取得所有履約價
    strikes = options_df['strike_price'].unique()
    
    max_pain_strike = spot_price
    min_total_value = float('inf')
    
    for strike in strikes:
        # 計算在此履約價下的總價值
        call_value = 0
        put_value = 0
        
        # Call 的價值
        calls = options_df[options_df['contract_type'] == 'call']
        for _, row in calls.iterrows():
            if strike > row['strike_price']:
                call_value += (strike - row['strike_price']) * row['open_interest']
        
        # Put 的價值
        puts = options_df[options_df['contract_type'] == 'put']
        for _, row in puts.iterrows():
            if strike < row['strike_price']:
                put_value += (row['strike_price'] - strike) * row['open_interest']
        
        total_value = call_value + put_value
        
        if total_value < min_total_value:
            min_total_value = total_value
            max_pain_strike = strike
    
    logger.info(f"最大痛點: {max_pain_strike}")
    
    return max_pain_strike


def calculate_iv_percentile(
    current_iv: float,
    iv_history: pd.Series,
    window: int = None
) -> float:
    """
    計算隱含波動率百分位
    
    Args:
        current_iv: 當前隱含波動率
        iv_history: 歷史隱含波動率序列
        window: 計算窗口
    
    Returns:
        IV 百分位 (0-100)
    """
    if window is None:
        window = FEATURE_CONFIG['volatility_metrics']['iv_percentile_window']
    
    recent_iv = iv_history.tail(window)
    
    if len(recent_iv) == 0:
        return 50.0
    
    percentile = (recent_iv < current_iv).sum() / len(recent_iv) * 100
    
    return percentile


def calculate_iv_hv_ratio(implied_vol: float, historical_vol: float) -> float:
    """
    計算 IV/HV 比值
    
    比值 < 1: 隱含波動率低於歷史波動率,適合買方進場
    比值 > 1: 隱含波動率高於歷史波動率,適合賣方進場
    
    Args:
        implied_vol: 隱含波動率
        historical_vol: 歷史波動率
    
    Returns:
        IV/HV 比值
    """
    if historical_vol == 0 or np.isnan(historical_vol):
        return np.nan
    
    return implied_vol / historical_vol


def calculate_volatility_smile(options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    計算波動率微笑 (Volatility Smile)
    
    Args:
        options_df: 選擇權資料 DataFrame
        spot_price: 當前現貨價格
    
    Returns:
        包含履約價、價內外程度與隱含波動率的 DataFrame
    """
    if options_df.empty or 'implied_volatility' not in options_df.columns:
        return pd.DataFrame()
    
    result = options_df.groupby('strike_price').agg({
        'implied_volatility': 'mean'
    }).reset_index()
    
    # 計算價內外程度 (Moneyness)
    result['moneyness'] = result['strike_price'] / spot_price
    
    # 排序
    result = result.sort_values('strike_price')
    
    return result


def analyze_options_chain(
    options_df: pd.DataFrame,
    spot_price: float,
    hv: float
) -> dict:
    """
    綜合分析選擇權鏈
    
    Args:
        options_df: 選擇權資料 DataFrame
        spot_price: 當前現貨價格
        hv: 歷史波動率
    
    Returns:
        分析結果字典
    """
    logger.info("分析選擇權鏈...")
    
    analysis = {}
    
    # Put/Call Ratio
    analysis['pcr_volume'] = calculate_put_call_ratio(options_df)
    analysis['pcr_oi'] = calculate_put_call_oi_ratio(options_df)
    
    # 最大痛點
    analysis['max_pain'] = calculate_max_pain(options_df, spot_price)
    
    # 平均隱含波動率
    if 'implied_volatility' in options_df.columns:
        avg_iv = options_df['implied_volatility'].mean()
        analysis['avg_iv'] = avg_iv
        analysis['iv_hv_ratio'] = calculate_iv_hv_ratio(avg_iv, hv)
    else:
        analysis['avg_iv'] = np.nan
        analysis['iv_hv_ratio'] = np.nan
    
    # 市場情緒判斷
    if not np.isnan(analysis['pcr_volume']):
        if analysis['pcr_volume'] > 1.2:
            analysis['sentiment'] = 'bearish'  # 看跌
        elif analysis['pcr_volume'] < 0.8:
            analysis['sentiment'] = 'bullish'  # 看漲
        else:
            analysis['sentiment'] = 'neutral'  # 中性
    else:
        analysis['sentiment'] = 'unknown'
    
    # 波動率環境
    if not np.isnan(analysis['iv_hv_ratio']):
        if analysis['iv_hv_ratio'] < 0.85:
            analysis['volatility_environment'] = 'low'  # 低波動,適合買方
        elif analysis['iv_hv_ratio'] > 1.15:
            analysis['volatility_environment'] = 'high'  # 高波動,適合賣方
        else:
            analysis['volatility_environment'] = 'normal'
    else:
        analysis['volatility_environment'] = 'unknown'
    
    logger.info(f"[OK] 選擇權鏈分析完成: {analysis}")
    
    return analysis


# 測試程式碼
if __name__ == "__main__":
    # 建立測試資料
    dates = pd.date_range('2024-01-01', periods=100)
    test_df = pd.DataFrame({
        'date': dates,
        'high': np.random.randn(100).cumsum() + 18100,
        'low': np.random.randn(100).cumsum() + 17900,
        'close': np.random.randn(100).cumsum() + 18000,
    })
    
    # 計算歷史波動率
    hv = calculate_historical_volatility(test_df)
    print(f"歷史波動率 (最新): {hv.iloc[-1]:.2%}")
    
    # 建立選擇權測試資料
    options_test = pd.DataFrame({
        'strike_price': [17800, 17900, 18000, 18100, 18200] * 2,
        'contract_type': ['call'] * 5 + ['put'] * 5,
        'volume': np.random.randint(100, 1000, 10),
        'open_interest': np.random.randint(500, 5000, 10),
        'implied_volatility': np.random.uniform(0.15, 0.25, 10)
    })
    
    # 分析選擇權鏈
    analysis = analyze_options_chain(options_test, 18000, hv.iloc[-1])
    print(f"\n選擇權鏈分析: {analysis}")
