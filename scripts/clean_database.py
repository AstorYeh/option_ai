"""
資料清洗腳本 - 清理資料庫中的異常資料
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.database import Database
from loguru import logger

def clean_database():
    """清理資料庫中的異常資料"""
    logger.info("=== 開始清理資料庫 ===")
    
    with Database() as db:
        # 讀取原始資料
        df = db.get_futures_data()
        logger.info(f"原始資料筆數: {len(df)}")
        
        if df.empty:
            logger.error("無資料")
            return
        
        # 1. 移除價格為 0 的資料
        before = len(df)
        df = df[
            (df['open'] > 0) & 
            (df['high'] > 0) & 
            (df['low'] > 0) & 
            (df['close'] > 0)
        ]
        removed = before - len(df)
        logger.info(f"移除價格為 0 的資料: {removed} 筆")
        
        # 2. 移除成交量為 0 的資料
        before = len(df)
        df = df[df['volume'] > 0]
        removed = before - len(df)
        logger.info(f"移除成交量為 0 的資料: {removed} 筆")
        
        # 3. 移除價格異常的資料 (使用 IQR 方法)
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        before = len(df)
        df = df[
            (df['close'] >= lower_bound) & 
            (df['close'] <= upper_bound)
        ]
        removed = before - len(df)
        logger.info(f"移除價格異常的資料: {removed} 筆")
        logger.info(f"價格範圍: {lower_bound:.0f} ~ {upper_bound:.0f}")
        
        # 4. 驗證價格邏輯
        before = len(df)
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        removed = before - len(df)
        logger.info(f"移除價格邏輯錯誤的資料: {removed} 筆")
        
        # 5. 移除重複資料
        before = len(df)
        df = df.drop_duplicates(subset=['date'], keep='last')
        removed = before - len(df)
        logger.info(f"移除重複資料: {removed} 筆")
        
        # 6. 排序
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"清理後資料筆數: {len(df)}")
        
        # 檢查資料庫欄位
        logger.info("檢查資料庫結構...")
        cursor = db.conn.execute("PRAGMA table_info(futures_daily)")
        columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"資料庫欄位: {columns}")
        
        # 確保 DataFrame 欄位與資料庫一致
        # 如果資料庫使用 max/min 而非 high/low,需要轉換
        if 'max' in columns and 'high' in df.columns:
            logger.info("轉換欄位名稱: high -> max, low -> min")
            df = df.rename(columns={'high': 'max', 'low': 'min'})
        
        # 7. 更新資料庫
        logger.info("更新資料庫...")
        
        # 刪除舊資料
        db.conn.execute("DELETE FROM futures_daily")
        db.conn.commit()
        logger.info("已清空舊資料")
        
        # 插入清理後的資料
        df.to_sql('futures_daily', db.conn, if_exists='append', index=False)
        db.conn.commit()
        logger.info(f"已插入 {len(df)} 筆清理後的資料")
        
        # 顯示統計
        logger.info("\n=== 清理後統計 ===")
        logger.info(f"日期範圍: {df['date'].min()} ~ {df['date'].max()}")
        
        # 使用正確的欄位名稱
        close_col = 'close'
        volume_col = 'volume'
        
        logger.info(f"收盤價範圍: {df[close_col].min():.0f} ~ {df[close_col].max():.0f}")
        logger.info(f"平均收盤價: {df[close_col].mean():.0f}")
        logger.info(f"平均成交量: {df[volume_col].mean():.0f}")
        
        # 顯示最近 5 筆資料
        logger.info("\n=== 最近 5 筆資料 ===")
        display_cols = ['date', 'open', 'close', 'volume']
        if 'max' in df.columns:
            display_cols.insert(2, 'max')
            display_cols.insert(3, 'min')
        else:
            display_cols.insert(2, 'high')
            display_cols.insert(3, 'low')
        
        print(df.tail(5)[display_cols])
        
        logger.info("\n[OK] 資料清理完成!")

if __name__ == "__main__":
    clean_database()
