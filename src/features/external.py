"""
外部因子特徵工程
整合美股、權值股、法人留倉等外部因子
"""
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/database/options.db")


def load_us_stocks(db_path=DB_PATH):
    """從 DB 載入美股資料"""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM us_stocks", conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.warning(f"載入美股資料失敗: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_tw_stocks(db_path=DB_PATH):
    """從 DB 載入台灣權值股資料"""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM tw_stocks", conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.warning(f"載入權值股資料失敗: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_institutional(db_path=DB_PATH):
    """從 DB 載入三大法人資料"""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM institutional_investors", conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.warning(f"載入法人資料失敗: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def build_us_features(us_df):
    """
    建立美股隔夜因子
    
    美股收盤 → 隔天台股開盤，因此用 shift 對齊
    台股 T 日的特徵 = 美股 T-1 日的收盤
    """
    if us_df.empty:
        logger.warning("[SKIP] 無美股資料")
        return pd.DataFrame()
    
    features = pd.DataFrame()
    
    for symbol in ['SPY', 'QQQ', 'SOXX']:
        sym_df = us_df[us_df['symbol'] == symbol].copy()
        sym_df = sym_df.sort_values('date').drop_duplicates('date')
        
        # 計算報酬率
        sym_df[f'{symbol.lower()}_return'] = sym_df['close'].pct_change()
        
        # 5 日動量
        sym_df[f'{symbol.lower()}_mom5'] = sym_df['close'].pct_change(5)
        
        # 波動率 (10 日)
        sym_df[f'{symbol.lower()}_vol10'] = sym_df[f'{symbol.lower()}_return'].rolling(10).std()
        
        cols_to_use = ['date', f'{symbol.lower()}_return', 
                       f'{symbol.lower()}_mom5', f'{symbol.lower()}_vol10']
        sym_features = sym_df[cols_to_use].set_index('date')
        
        if features.empty:
            features = sym_features
        else:
            features = features.join(sym_features, how='outer')
    
    # 綜合美股情緒
    if 'spy_return' in features.columns:
        features['us_sentiment'] = (
            features.get('spy_return', 0) * 0.4 +
            features.get('qqq_return', 0) * 0.3 +
            features.get('soxx_return', 0) * 0.3
        )
        
        # VIX 代理: 用 SPY 波動率
        features['us_risk'] = features.get('spy_vol10', 0)
    
    features = features.reset_index().rename(columns={'index': 'date'})
    
    logger.info(f"[OK] 美股特徵: {len(features)} 筆, {len(features.columns)-1} 個指標")
    return features


def build_tw_stock_features(tw_df):
    """
    建立台灣權值股因子
    
    台積電單獨計算 + 前五大總體指標
    """
    if tw_df.empty:
        logger.warning("[SKIP] 無權值股資料")
        return pd.DataFrame()
    
    features = pd.DataFrame()

    # 確認 close 欄位名稱
    close_col = 'close' if 'close' in tw_df.columns else 'Close'
    volume_col = 'Trading_Volume' if 'Trading_Volume' in tw_df.columns else 'volume'
    
    all_symbols = tw_df['symbol' if 'symbol' in tw_df.columns else 'stock_id'].unique()
    
    # 各股票的每日報酬
    daily_returns = pd.DataFrame()
    
    for symbol in all_symbols:
        sym_col = 'symbol' if 'symbol' in tw_df.columns else 'stock_id'
        sym_df = tw_df[tw_df[sym_col] == symbol].copy()
        sym_df = sym_df.sort_values('date').drop_duplicates('date')
        
        sym_df[f'{symbol}_return'] = sym_df[close_col].pct_change()
        
        sym_features = sym_df[['date', f'{symbol}_return']].set_index('date')
        
        if daily_returns.empty:
            daily_returns = sym_features
        else:
            daily_returns = daily_returns.join(sym_features, how='outer')
    
    # 台積電專屬指標
    tsmc_df = tw_df[tw_df[sym_col] == '2330'].copy().sort_values('date').drop_duplicates('date')
    if not tsmc_df.empty:
        tsmc_df['tsmc_return'] = tsmc_df[close_col].pct_change()
        tsmc_df['tsmc_mom5'] = tsmc_df[close_col].pct_change(5)
        
        # 台積電量能比 (vs 20MA)
        tsmc_df['tsmc_vol_ratio'] = (
            tsmc_df[volume_col] / tsmc_df[volume_col].rolling(20).mean()
        )
        
        features = tsmc_df[['date', 'tsmc_return', 'tsmc_mom5', 'tsmc_vol_ratio']].set_index('date')
    
    # 前五大權值股綜合指標
    return_cols = [c for c in daily_returns.columns if c.endswith('_return')]
    if return_cols:
        daily_returns['top5_avg_return'] = daily_returns[return_cols].mean(axis=1)
        daily_returns['top5_mom5'] = daily_returns['top5_avg_return'].rolling(5).sum()
        daily_returns['sector_dispersion'] = daily_returns[return_cols].std(axis=1)
        
        agg_features = daily_returns[['top5_avg_return', 'top5_mom5', 'sector_dispersion']]
        
        if features.empty:
            features = agg_features
        else:
            features = features.join(agg_features, how='outer')
    
    # 台積電相對大盤強弱
    if 'tsmc_return' in features.columns and 'top5_avg_return' in features.columns:
        features['tsmc_vs_market'] = features['tsmc_return'] - features['top5_avg_return']
    
    features = features.reset_index().rename(columns={'index': 'date'})
    
    logger.info(f"[OK] 權值股特徵: {len(features)} 筆, {len(features.columns)-1} 個指標")
    return features


def build_institutional_features(inst_df):
    """
    建立法人留倉因子
    
    依每日分組，計算法人淨留倉合計
    """
    if inst_df.empty:
        logger.warning("[SKIP] 無法人資料")
        return pd.DataFrame()
    
    long_col = 'long_open_interest_balance_volume'
    short_col = 'short_open_interest_balance_volume'
    
    if long_col not in inst_df.columns or short_col not in inst_df.columns:
        logger.warning("[SKIP] 法人資料缺少留倉欄位")
        return pd.DataFrame()
    
    # 轉數值
    inst_df[long_col] = pd.to_numeric(inst_df[long_col], errors='coerce').fillna(0)
    inst_df[short_col] = pd.to_numeric(inst_df[short_col], errors='coerce').fillna(0)
    inst_df['net_oi'] = inst_df[long_col] - inst_df[short_col]
    
    # 按日期取得投資人名稱列表
    investor_types = inst_df['institutional_investors'].unique()
    
    features = pd.DataFrame()
    
    for i, inv_type in enumerate(investor_types):
        inv_df = inst_df[inst_df['institutional_investors'] == inv_type].copy()
        inv_df = inv_df.sort_values('date').drop_duplicates('date')
        
        prefix = f'inst{i}'
        inv_df[f'{prefix}_net_oi'] = inv_df['net_oi']
        inv_df[f'{prefix}_oi_change'] = inv_df['net_oi'].diff()
        
        inv_features = inv_df[['date', f'{prefix}_net_oi', f'{prefix}_oi_change']].set_index('date')
        
        if features.empty:
            features = inv_features
        else:
            features = features.join(inv_features, how='outer')
    
    # 法人一致性 (前兩類法人是否同方向)
    oi_cols = [c for c in features.columns if c.endswith('_net_oi')]
    if len(oi_cols) >= 2:
        features['inst_consensus'] = np.sign(
            features[oi_cols[0]]
        ) * np.sign(
            features[oi_cols[1]]
        )
    
    # 全體法人淨留倉
    features['inst_total_net_oi'] = features[oi_cols].sum(axis=1) if oi_cols else 0
    features['inst_total_oi_change'] = features['inst_total_net_oi'].diff()
    
    features = features.reset_index().rename(columns={'index': 'date'})
    
    logger.info(f"[OK] 法人特徵: {len(features)} 筆, {len(features.columns)-1} 個指標")
    return features


def add_external_features(df, db_path=DB_PATH):
    """
    主函數: 將所有外部特徵合併到主 DataFrame
    
    Args:
        df: 台指期 DataFrame (需含 'date' 欄位)
        db_path: 資料庫路徑
    
    Returns:
        加入外部特徵的 DataFrame
    """
    logger.info("=== 計算外部因子特徵 ===")
    
    result = df.copy()
    
    # 確保 date 為 datetime
    result['date'] = pd.to_datetime(result['date'])
    
    initial_cols = len(result.columns)
    
    # 1. 美股特徵
    us_df = load_us_stocks(db_path)
    if not us_df.empty:
        us_features = build_us_features(us_df)
        if not us_features.empty:
            us_features['date'] = pd.to_datetime(us_features['date'])
            # 美股 T-1 → 台股 T (shift 1 個交易日)
            us_dates = us_features['date'].sort_values().unique()
            tw_dates = result['date'].sort_values().unique()
            
            # 建立日期映射: 美股日期 → 下一個台股交易日
            date_map = {}
            tw_idx = 0
            for us_date in sorted(us_dates):
                while tw_idx < len(tw_dates) and tw_dates[tw_idx] <= us_date:
                    tw_idx += 1
                if tw_idx < len(tw_dates):
                    date_map[us_date] = tw_dates[tw_idx]
            
            us_features['tw_date'] = us_features['date'].map(
                lambda d: date_map.get(pd.Timestamp(d))
            )
            us_features = us_features.dropna(subset=['tw_date'])
            us_features = us_features.drop(columns=['date']).rename(
                columns={'tw_date': 'date'}
            )
            us_features = us_features.drop_duplicates('date')
            
            result = result.merge(us_features, on='date', how='left')
    
    # 2. 權值股特徵 (同日)
    tw_df = load_tw_stocks(db_path)
    if not tw_df.empty:
        tw_features = build_tw_stock_features(tw_df)
        if not tw_features.empty:
            tw_features['date'] = pd.to_datetime(tw_features['date'])
            tw_features = tw_features.drop_duplicates('date')
            result = result.merge(tw_features, on='date', how='left')
    
    # 3. 法人留倉特徵 (同日)
    inst_df = load_institutional(db_path)
    if not inst_df.empty:
        inst_features = build_institutional_features(inst_df)
        if not inst_features.empty:
            inst_features['date'] = pd.to_datetime(inst_features['date'])
            inst_features = inst_features.drop_duplicates('date')
            result = result.merge(inst_features, on='date', how='left')
    
    # 4. 跨市場關聯因子
    if 'spy_return' in result.columns and 'close' in result.columns:
        # 台指期報酬率
        tw_return = result['close'].pct_change()
        
        # 美台相關性 (20 日滾動)
        result['us_tw_corr_20d'] = tw_return.rolling(20).corr(
            result['spy_return'].fillna(0)
        )
        
        # 隔夜跳空預測 (美股大跌 → 可能跳空)
        result['overnight_gap_signal'] = (
            result['us_sentiment'].fillna(0) * 
            result.get('us_tw_corr_20d', pd.Series(0.5, index=result.index)).fillna(0.5)
        )
        
        # 全球風險評分
        result['global_risk_score'] = (
            result.get('spy_vol10', pd.Series(0, index=result.index)).fillna(0) * 100
        )
    
    # 填充缺失值
    new_cols = [c for c in result.columns if c not in df.columns]
    for col in new_cols:
        result[col] = result[col].ffill().bfill().fillna(0)
    
    added = len(result.columns) - initial_cols
    logger.info(f"[OK] 外部因子計算完成, 新增 {added} 個特徵")
    
    return result
