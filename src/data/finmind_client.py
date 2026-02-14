"""
FinMind API 客戶端
用於抓取台指期與選擇權資料
"""
import requests
import pandas as pd
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from config.api_config import (
    FINMIND_API_TOKEN,
    FINMIND_API_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    FINMIND_REQUEST_DELAY
)
from config.settings import FUTURES_SYMBOL, OPTIONS_SYMBOL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinMindClient:
    """FinMind API 客戶端"""
    
    def __init__(self, api_token: str = None):
        """
        初始化客戶端
        
        Args:
            api_token: API Token (若未提供則使用環境變數)
        """
        self.api_token = api_token or FINMIND_API_TOKEN
        self.api_url = FINMIND_API_URL
        
        if not self.api_token:
            logger.warning("未設定 FinMind API Token,將使用免費版限制")
    
    def _make_request(self, params: Dict[str, Any], retries: int = 0) -> Optional[pd.DataFrame]:
        """
        發送 API 請求
        
        Args:
            params: 請求參數
            retries: 當前重試次數
        
        Returns:
            資料 DataFrame 或 None
        """
        try:
            # 加入 API Token
            if self.api_token:
                params['token'] = self.api_token
            
            logger.debug(f"發送 API 請求: {params}")
            
            response = requests.get(
                self.api_url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 檢查回應狀態
            if data.get('status') != 200:
                logger.error(f"API 回應錯誤: {data.get('msg', 'Unknown error')}")
                return None
            
            # 轉換為 DataFrame
            df = pd.DataFrame(data['data'])
            
            if df.empty:
                logger.warning(f"API 回應資料為空: {params}")
                return None
            
            logger.info(f"成功取得 {len(df)} 筆資料")
            
            # 請求間隔(避免超過 API 限制)
            time.sleep(FINMIND_REQUEST_DELAY)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API 請求失敗: {e}")
            
            # 重試機制
            if retries < MAX_RETRIES:
                logger.info(f"等待 {RETRY_DELAY} 秒後重試... ({retries + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return self._make_request(params, retries + 1)
            else:
                logger.error(f"達到最大重試次數,放棄請求")
                return None
        
        except Exception as e:
            logger.error(f"處理 API 回應時發生錯誤: {e}")
            return None
    
    def get_futures_data(
        self,
        start_date: str,
        end_date: str,
        symbol: str = FUTURES_SYMBOL
    ) -> Optional[pd.DataFrame]:
        """
        取得台指期歷史資料
        
        Args:
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            symbol: 期貨代碼 (預設: TX)
        
        Returns:
            包含 OHLCV 的 DataFrame
        """
        logger.info(f"取得台指期資料: {start_date} ~ {end_date}")
        
        params = {
            'dataset': 'TaiwanFuturesDaily',
            'data_id': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        
        df = self._make_request(params)
        
        if df is not None and not df.empty:
            # 資料清理與轉換
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 轉換數值欄位
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"台指期資料處理完成: {len(df)} 筆")
        
        return df
    
    def get_options_data(
        self,
        date: str,
        symbol: str = OPTIONS_SYMBOL
    ) -> Optional[pd.DataFrame]:
        """
        取得選擇權每日報價資料
        
        Args:
            date: 日期 (YYYY-MM-DD)
            symbol: 選擇權代碼 (預設: TXO)
        
        Returns:
            選擇權報價 DataFrame
        """
        logger.info(f"取得選擇權資料: {date}")
        
        params = {
            'dataset': 'TaiwanOptionDaily',
            'data_id': symbol,
            'start_date': date,
            'end_date': date
        }
        
        df = self._make_request(params)
        
        if df is not None and not df.empty:
            # 資料清理與轉換
            df['date'] = pd.to_datetime(df['date'])
            
            # 轉換數值欄位
            numeric_cols = ['strike_price', 'open', 'high', 'low', 'close', 
                          'volume', 'open_interest']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 分離 Call 與 Put
            if 'contract_type' in df.columns:
                df['contract_type'] = df['contract_type'].str.lower()
            
            logger.info(f"選擇權資料處理完成: {len(df)} 筆")
        
        return df
    
    def get_options_data_range(
        self,
        start_date: str,
        end_date: str,
        symbol: str = OPTIONS_SYMBOL
    ) -> Optional[pd.DataFrame]:
        """
        取得選擇權歷史資料(日期範圍)
        
        Args:
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            symbol: 選擇權代碼
        
        Returns:
            選擇權歷史資料 DataFrame
        """
        logger.info(f"取得選擇權歷史資料: {start_date} ~ {end_date}")
        
        # 由於選擇權資料量大,建議分批取得
        all_data = []
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            df = self.get_options_data(date_str, symbol)
            
            if df is not None and not df.empty:
                all_data.append(df)
            
            current_date += timedelta(days=1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"選擇權歷史資料處理完成: {len(result)} 筆")
            return result
        else:
            logger.warning("未取得任何選擇權歷史資料")
            return None
    
    def get_institutional_investors(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        取得三大法人期貨留倉資料
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
        
        Returns:
            法人留倉資料 DataFrame
        """
        logger.info(f"取得法人留倉資料: {start_date} ~ {end_date}")
        
        params = {
            'dataset': 'TaiwanFuturesInstitutionalInvestors',
            'start_date': start_date,
            'end_date': end_date
        }
        
        df = self._make_request(params)
        
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"法人留倉資料處理完成: {len(df)} 筆")
        
        return df
    
    def test_connection(self) -> bool:
        """測試 API 連線"""
        logger.info("測試 FinMind API 連線...")
        
        try:
            # 使用較早的日期測試(避免查詢未來日期)
            end_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            data = self.get_futures_data(start_date=start_date, end_date=end_date)
            
            if data is not None and not data.empty: # Added 'is not None' for robustness
                logger.info(f"[OK] FinMind API 連線成功,取得 {len(data)} 筆資料")
                return True
            else:
                logger.error("[ERROR] FinMind API 連線失敗")
                return False
        except Exception as e:
            logger.error(f"[ERROR] FinMind API 連線失敗: {e}")
            return False


# 測試程式碼
if __name__ == "__main__":
    client = FinMindClient()
    
    # 測試連線
    if client.test_connection():
        # 取得最近 5 天的台指期資料
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        print(f"\n[DATA] 台指期資料 ({start_date} ~ {end_date}):")
        df_futures = client.get_futures_data(start_date, end_date)
        if df_futures is not None:
            print(df_futures.head())
        
        print(f"\n[DATA] 選擇權資料 ({end_date}):")
        df_options = client.get_options_data(end_date)
        if df_options is not None:
            print(df_options.head(10))
