"""
外部因子資料下載器
下載美股、權值股、法人留倉等外部因子
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
from loguru import logger

from src.data.finmind_client import FinMindClient
from config.settings import DATA_DIR

# ===== 設定 =====
DB_PATH = Path(DATA_DIR) / "database" / "options.db"

# 美股 ETF (作為指數代理)
US_SYMBOLS = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    'SOXX': '費城半導體',
}

# 台灣權值股
TW_SYMBOLS = {
    '2330': '台積電',
    '2317': '鴻海',
    '2454': '聯發科',
    '2308': '台達電',
    '2881': '富邦金',
}


def download_us_stocks(client, start_date, end_date):
    """下載美股 ETF 日線資料"""
    logger.info("=== 下載美股資料 ===")
    all_data = []
    
    for symbol, name in US_SYMBOLS.items():
        logger.info(f"  下載 {symbol} ({name})...")
        params = {
            'dataset': 'USStockPrice',
            'data_id': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        df = client._make_request(params)
        
        if df is not None and not df.empty:
            df['symbol'] = symbol
            df['name'] = name
            df['date'] = pd.to_datetime(df['date'])
            # 統一欄位名稱
            rename_map = {'Close': 'close', 'Open': 'open', 'High': 'high', 
                         'Low': 'low', 'Volume': 'volume'}
            df = df.rename(columns=rename_map)
            all_data.append(df)
            logger.info(f"    [OK] {symbol}: {len(df)} 筆")
        else:
            logger.warning(f"    [WARN] {symbol}: 無資料")
        
        time.sleep(1)  # API 請求間隔
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"[OK] 美股總計: {len(result)} 筆")
        return result
    return pd.DataFrame()


def download_tw_stocks(client, start_date, end_date):
    """下載台灣權值股日線資料"""
    logger.info("=== 下載台灣權值股資料 ===")
    all_data = []
    
    for symbol, name in TW_SYMBOLS.items():
        logger.info(f"  下載 {symbol} ({name})...")
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        df = client._make_request(params)
        
        if df is not None and not df.empty:
            df['symbol'] = symbol
            df['name'] = name
            df['date'] = pd.to_datetime(df['date'])
            all_data.append(df)
            logger.info(f"    [OK] {symbol}: {len(df)} 筆")
        else:
            logger.warning(f"    [WARN] {symbol}: 無資料")
        
        time.sleep(1)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"[OK] 權值股總計: {len(result)} 筆")
        return result
    return pd.DataFrame()


def download_institutional(client, start_date, end_date):
    """下載三大法人期貨留倉資料"""
    logger.info("=== 下載三大法人留倉 ===")
    
    params = {
        'dataset': 'TaiwanFuturesInstitutionalInvestors',
        'data_id': 'TX',
        'start_date': start_date,
        'end_date': end_date
    }
    df = client._make_request(params)
    
    if df is not None and not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"[OK] 法人留倉: {len(df)} 筆")
        return df
    
    logger.warning("[WARN] 法人留倉: 無資料")
    return pd.DataFrame()


def save_to_db(df, table_name, db_path=DB_PATH):
    """存入 SQLite"""
    if df.empty:
        logger.warning(f"[SKIP] {table_name}: 無資料可存")
        return
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    
    # 日期轉字串
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    
    # 先刪除已有資料（避免重複）
    try:
        existing = pd.read_sql(f"SELECT DISTINCT date FROM {table_name}", conn)
        if not existing.empty:
            new_dates = set(df['date'].unique()) - set(existing['date'].unique())
            if new_dates:
                df = df[df['date'].isin(new_dates)]
                logger.info(f"  新增 {len(df)} 筆 (排除已存在日期)")
            else:
                logger.info(f"  [SKIP] 所有日期已存在")
                conn.close()
                return
    except Exception:
        pass  # 表不存在，全部寫入
    
    df.to_sql(table_name, conn, if_exists='append', index=False)
    
    total = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn)
    logger.info(f"[OK] {table_name}: 已存入, 總計 {total['cnt'].iloc[0]} 筆")
    conn.close()


def download_all():
    """下載所有外部因子"""
    logger.info("=" * 60)
    logger.info("=== 外部因子資料下載器 ===")
    logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    client = FinMindClient()
    
    # 日期範圍 (與台指期一致: 3 年)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    logger.info(f"日期範圍: {start_date} ~ {end_date}")
    
    # 1. 美股
    us_df = download_us_stocks(client, start_date, end_date)
    save_to_db(us_df, 'us_stocks')
    
    # 2. 權值股
    tw_df = download_tw_stocks(client, start_date, end_date)
    save_to_db(tw_df, 'tw_stocks')
    
    # 3. 法人留倉
    inst_df = download_institutional(client, start_date, end_date)
    save_to_db(inst_df, 'institutional_investors')
    
    logger.info("\n" + "=" * 60)
    logger.info("[OK] 外部因子下載完畢!")
    logger.info("=" * 60)
    
    return {
        'us_stocks': len(us_df),
        'tw_stocks': len(tw_df),
        'institutional': len(inst_df),
    }


if __name__ == "__main__":
    try:
        result = download_all()
        print(f"\n下載結果:")
        for k, v in result.items():
            print(f"  {k}: {v} 筆")
    except Exception as e:
        logger.error(f"下載失敗: {e}")
        import traceback
        traceback.print_exc()
