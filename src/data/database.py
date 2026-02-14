"""
資料庫操作模組
使用 SQLite 儲存歷史資料
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from config.settings import DATABASE_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """資料庫操作類別"""
    
    def __init__(self, db_path: str = None):
        """
        初始化資料庫連線
        
        Args:
            db_path: 資料庫路徑
        """
        self.db_path = db_path or DATABASE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.cursor = None
        
        logger.info(f"資料庫路徑: {self.db_path}")
    
    def connect(self):
        """建立資料庫連線"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info("資料庫連線成功")
        except sqlite3.Error as e:
            logger.error(f"資料庫連線失敗: {e}")
            raise
    
    def close(self):
        """關閉資料庫連線"""
        if self.conn:
            self.conn.close()
            logger.info("資料庫連線已關閉")
    
    def __enter__(self):
        """Context manager 進入"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出"""
        self.close()
    
    def create_tables(self):
        """建立資料表"""
        logger.info("建立資料表...")
        
        # 不預先建立結構,讓 pandas 自動建立
        # 只建立索引以提升查詢效能
        
        self.conn.commit()
        logger.info("[OK] 資料表建立完成")
    
    def insert_futures_data(self, df: pd.DataFrame) -> bool:
        """
        插入台指期資料
        
        Args:
            df: 包含台指期資料的 DataFrame
        
        Returns:
            是否成功
        """
        try:
            # 直接儲存所有欄位
            df.to_sql('futures_daily', self.conn, if_exists='append', index=False)
            self.conn.commit()
            logger.info(f"成功插入 {len(df)} 筆台指期資料")
            return True
        except Exception as e:
            logger.error(f"插入台指期資料失敗: {e}")
            self.conn.rollback()
            return False
    
    def insert_options_data(self, df: pd.DataFrame):
        """
        插入選擇權資料
        
        Args:
            df: 包含選擇權資料的 DataFrame
        """
        try:
            df.to_sql('options_daily', self.conn, if_exists='append', index=False)
            self.conn.commit()
            logger.info(f"[OK] 插入 {len(df)} 筆選擇權資料")
        except sqlite3.IntegrityError:
            logger.warning("部分選擇權資料已存在,跳過")
        except Exception as e:
            logger.error(f"插入選擇權資料失敗: {e}")
            raise
    
    def get_futures_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        查詢台指期資料
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
        
        Returns:
            台指期資料 DataFrame
        """
        query = "SELECT * FROM futures_daily"
        conditions = []
        
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        
        # 重新命名欄位以符合標準名稱
        column_mapping = {
            'max': 'high',
            'min': 'low'
        }
        df = df.rename(columns=column_mapping)
        
        logger.info(f"查詢到 {len(df)} 筆台指期資料")
        return df
    
    def get_options_data(
        self,
        date: str,
        contract_type: str = None
    ) -> pd.DataFrame:
        """
        查詢選擇權資料
        
        Args:
            date: 日期
            contract_type: 契約類型 ('call' 或 'put')
        
        Returns:
            選擇權資料 DataFrame
        """
        query = f"SELECT * FROM options_daily WHERE date = '{date}'"
        
        if contract_type:
            query += f" AND contract_type = '{contract_type}'"
        
        query += " ORDER BY strike_price"
        
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"查詢到 {len(df)} 筆選擇權資料")
        return df
    
    def get_latest_futures_date(self) -> Optional[str]:
        """取得最新的台指期資料日期"""
        query = "SELECT MAX(date) as latest_date FROM futures_daily"
        result = pd.read_sql_query(query, self.conn)
        
        if not result.empty and result['latest_date'][0]:
            return result['latest_date'][0]
        return None
    
    def insert_prediction(
        self,
        date: str,
        prediction_type: str,
        direction: str,
        confidence: float,
        suggested_strike: float = None
    ):
        """
        插入預測記錄
        
        Args:
            date: 預測日期
            prediction_type: 預測類型 ('direction' 或 'volatility')
            direction: 方向 ('buy_call', 'buy_put', 'hold')
            confidence: 信心度 (0-1)
            suggested_strike: 建議履約價
        """
        self.cursor.execute("""
            INSERT INTO predictions (date, prediction_type, direction, confidence, suggested_strike)
            VALUES (?, ?, ?, ?, ?)
        """, (date, prediction_type, direction, confidence, suggested_strike))
        
        self.conn.commit()
        logger.info(f"[OK] 插入預測記錄: {date} - {direction}")
    
    def insert_trade(
        self,
        entry_date: str,
        exit_date: str,
        contract_type: str,
        strike_price: float,
        entry_price: float,
        exit_price: float,
        quantity: int
    ):
        """
        插入交易記錄
        
        Args:
            entry_date: 進場日期
            exit_date: 出場日期
            contract_type: 契約類型
            strike_price: 履約價
            entry_price: 進場價格(權利金)
            exit_price: 出場價格
            quantity: 數量
        """
        profit_loss = (exit_price - entry_price) * quantity * 50  # 台指選擇權每點 50 元
        return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        self.cursor.execute("""
            INSERT INTO trades (entry_date, exit_date, contract_type, strike_price,
                              entry_price, exit_price, quantity, profit_loss, return_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (entry_date, exit_date, contract_type, strike_price, entry_price,
              exit_price, quantity, profit_loss, return_pct))
        
        self.conn.commit()
        logger.info(f"[OK] 插入交易記錄: {entry_date} - 損益 {profit_loss:.0f}")
    
    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """
        取得交易歷史
        
        Args:
            limit: 筆數限制
        
        Returns:
            交易歷史 DataFrame
        """
        query = f"""
            SELECT * FROM trades
            ORDER BY entry_date DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df


# 測試程式碼
if __name__ == "__main__":
    with Database() as db:
        # 建立資料表
        db.create_tables()
        
        # 測試插入資料
        test_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'open': [18000, 18100],
            'high': [18100, 18200],
            'low': [17900, 18000],
            'close': [18050, 18150],
            'volume': [100000, 120000],
            'open_interest': [50000, 55000]
        })
        
        db.insert_futures_data(test_data)
        
        # 測試查詢
        result = db.get_futures_data('2024-01-01', '2024-01-02')
        print(result)
