"""
資料抓取與更新腳本
每日自動抓取台指期與選擇權資料
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timedelta
from src.data.finmind_client import FinMindClient
from src.data.database import Database
from src.utils.logger import get_logger
from src.utils.helpers import get_taiwan_time, get_last_trading_day

logger = get_logger(__name__)


def fetch_futures_data(client: FinMindClient, db: Database, start_date: str, end_date: str):
    """
    抓取台指期資料
    
    Args:
        client: FinMind 客戶端
        db: 資料庫實例
        start_date: 開始日期
        end_date: 結束日期
    """
    logger.info(f"抓取台指期資料: {start_date} ~ {end_date}")
    
    df = client.get_futures_data(start_date, end_date)
    
    if df is not None and not df.empty:
        # 儲存到資料庫
        db.insert_futures_data(df)
        logger.info(f"[OK] 台指期資料已儲存: {len(df)} 筆")
    else:
        logger.warning("[WARN] 未取得台指期資料")


def fetch_options_data(client: FinMindClient, db: Database, date: str):
    """
    抓取選擇權資料
    
    Args:
        client: FinMind 客戶端
        db: 資料庫實例
        date: 日期
    """
    logger.info(f"抓取選擇權資料: {date}")
    
    df = client.get_options_data(date)
    
    if df is not None and not df.empty:
        # 儲存到資料庫
        db.insert_options_data(df)
        logger.info(f"[OK] 選擇權資料已儲存: {len(df)} 筆")
    else:
        logger.warning("[WARN] 未取得選擇權資料")


def initial_data_fetch(days: int = 365):
    """
    初始化資料抓取(首次使用)
    
    Args:
        days: 抓取天數
    """
    logger.info(f"開始初始化資料抓取(過去 {days} 天)...")
    
    client = FinMindClient()
    
    # 測試連線
    if not client.test_connection():
        logger.error("[ERROR] FinMind API 連線失敗,請檢查設定")
        return
    
    with Database() as db:
        # 建立資料表
        db.create_tables()
        
        # 計算日期範圍
        end_date = get_last_trading_day()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 抓取台指期資料
        fetch_futures_data(client, db, start_str, end_str)
        
        # 抓取選擇權資料(最近 30 天)
        logger.info("抓取選擇權資料(最近 30 天)...")
        current_date = end_date - timedelta(days=30)
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 只抓取交易日
                date_str = current_date.strftime('%Y-%m-%d')
                fetch_options_data(client, db, date_str)
            
            current_date += timedelta(days=1)
    
    logger.info("[OK] 初始化資料抓取完成!")


def daily_update():
    """每日資料更新"""
    logger.info("開始每日資料更新...")
    
    client = FinMindClient()
    
    with Database() as db:
        # 取得最新資料日期
        latest_date = db.get_latest_futures_date()
        
        if latest_date:
            logger.info(f"資料庫最新日期: {latest_date}")
            start_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # 若資料庫為空,抓取最近 7 天
            start_date = (get_last_trading_day() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        end_date = get_last_trading_day().strftime('%Y-%m-%d')
        
        # 抓取台指期資料
        fetch_futures_data(client, db, start_date, end_date)
        
        # 抓取今日選擇權資料
        fetch_options_data(client, db, end_date)
    
    logger.info("[OK] 每日資料更新完成!")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='台指期選擇權資料抓取工具')
    parser.add_argument(
        '--initial',
        action='store_true',
        help='初始化資料抓取(首次使用)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='初始化抓取天數(預設: 365)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.initial:
            initial_data_fetch(args.days)
        else:
            daily_update()
    except Exception as e:
        logger.error(f"[ERROR] 資料更新失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
