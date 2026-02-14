"""
下載完整歷史資料 (90 天)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.finmind_client import FinMindClient
from src.data.database import Database
from loguru import logger
import time

def download_historical_data(days=90):
    """下載歷史資料"""
    logger.info(f"=== 開始下載 {days} 天歷史資料 ===")
    
    # 計算日期範圍
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"日期範圍: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    # 初始化客戶端
    client = FinMindClient()
    
    # 測試連線
    if not client.test_connection():
        logger.error("API 連線失敗")
        return False
    
    # 下載資料
    logger.info("下載台指期資料...")
    futures_data = client.get_futures_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if futures_data is None or futures_data.empty:
        logger.error("下載失敗")
        return False
    
    logger.info(f"下載成功: {len(futures_data)} 筆資料")
    
    # 儲存到資料庫
    with Database() as db:
        # 清空舊資料
        logger.info("清空舊資料...")
        db.conn.execute("DELETE FROM futures_daily")
        db.conn.commit()
        
        # 插入新資料
        logger.info("插入新資料...")
        success = db.insert_futures_data(futures_data)
        
        if success:
            logger.info(f"[OK] 成功儲存 {len(futures_data)} 筆資料")
            
            # 顯示統計
            logger.info("\n=== 資料統計 ===")
            logger.info(f"日期範圍: {futures_data['date'].min()} ~ {futures_data['date'].max()}")
            logger.info(f"總筆數: {len(futures_data)}")
            
            # 顯示最近 5 筆
            logger.info("\n=== 最近 5 筆資料 ===")
            print(futures_data.tail(5)[['date', 'open', 'close', 'volume']])
            
            return True
        else:
            logger.error("儲存失敗")
            return False

if __name__ == "__main__":
    success = download_historical_data(days=90)
    
    if success:
        print("\n" + "="*60)
        print("[OK] 歷史資料下載完成!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[ERROR] 歷史資料下載失敗!")
        print("="*60)
