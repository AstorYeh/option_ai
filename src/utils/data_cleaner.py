"""
資料清洗與驗證模組
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """資料清洗器"""
    
    def __init__(self):
        """初始化資料清洗器"""
        self.cleaning_stats = {
            'total_rows': 0,
            'removed_duplicates': 0,
            'removed_nulls': 0,
            'removed_outliers': 0,
            'filled_missing': 0
        }
    
    def clean_futures_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗台指期資料
        
        Args:
            df: 原始資料
        
        Returns:
            清洗後的資料
        """
        logger.info("開始清洗台指期資料...")
        
        self.cleaning_stats['total_rows'] = len(df)
        original_count = len(df)
        
        # 1. 移除重複資料
        df = self._remove_duplicates(df)
        
        # 2. 處理缺失值
        df = self._handle_missing_values(df)
        
        # 3. 移除異常值
        df = self._remove_outliers(df)
        
        # 4. 驗證資料完整性
        df = self._validate_data(df)
        
        # 5. 排序資料
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"[OK] 資料清洗完成")
        logger.info(f"   原始筆數: {original_count}")
        logger.info(f"   清洗後筆數: {len(df)}")
        logger.info(f"   移除重複: {self.cleaning_stats['removed_duplicates']}")
        logger.info(f"   移除空值: {self.cleaning_stats['removed_nulls']}")
        logger.info(f"   移除異常: {self.cleaning_stats['removed_outliers']}")
        logger.info(f"   填補缺失: {self.cleaning_stats['filled_missing']}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除重複資料"""
        before = len(df)
        df = df.drop_duplicates(subset=['date'], keep='last')
        after = len(df)
        
        removed = before - after
        self.cleaning_stats['removed_duplicates'] = removed
        
        if removed > 0:
            logger.info(f"移除 {removed} 筆重複資料")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理缺失值"""
        # 檢查缺失值
        null_counts = df.isnull().sum()
        
        if null_counts.sum() == 0:
            logger.info("無缺失值")
            return df
        
        logger.info("處理缺失值...")
        
        # 移除關鍵欄位有缺失的資料
        critical_columns = ['date', 'close']
        before = len(df)
        df = df.dropna(subset=critical_columns)
        after = len(df)
        
        removed = before - after
        self.cleaning_stats['removed_nulls'] = removed
        
        if removed > 0:
            logger.warning(f"移除 {removed} 筆關鍵欄位缺失的資料")
        
        # 填補非關鍵欄位的缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                # 使用前向填充
                df[col] = df[col].fillna(method='ffill')
                # 如果還有缺失,使用後向填充
                df[col] = df[col].fillna(method='bfill')
                # 如果還有缺失,使用平均值
                df[col] = df[col].fillna(df[col].mean())
                
                self.cleaning_stats['filled_missing'] += df[col].isnull().sum()
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除異常值"""
        logger.info("檢測異常值...")
        
        before = len(df)
        
        # 檢測價格異常 (使用 IQR 方法)
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # 使用 3 倍 IQR (較寬鬆)
            upper_bound = Q3 + 3 * IQR
            
            # 標記異常值
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            if outliers.any():
                logger.warning(f"{col} 發現 {outliers.sum()} 個異常值")
                # 移除異常值
                df = df[~outliers]
        
        # 檢測成交量異常
        if 'volume' in df.columns:
            # 成交量不應為 0 或負數
            df = df[df['volume'] > 0]
        
        after = len(df)
        removed = before - after
        self.cleaning_stats['removed_outliers'] = removed
        
        if removed > 0:
            logger.info(f"移除 {removed} 筆異常資料")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """驗證資料完整性"""
        logger.info("驗證資料完整性...")
        
        before = len(df)
        
        # 驗證價格邏輯
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # high 應該是最高價
            invalid_high = (df['high'] < df['open']) | \
                          (df['high'] < df['close']) | \
                          (df['high'] < df['low'])
            
            # low 應該是最低價
            invalid_low = (df['low'] > df['open']) | \
                         (df['low'] > df['close']) | \
                         (df['low'] > df['high'])
            
            invalid = invalid_high | invalid_low
            
            if invalid.any():
                logger.warning(f"發現 {invalid.sum()} 筆價格邏輯錯誤的資料")
                df = df[~invalid]
        
        after = len(df)
        removed = before - after
        
        if removed > 0:
            logger.info(f"移除 {removed} 筆驗證失敗的資料")
        
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        檢查資料品質
        
        Args:
            df: 資料
        
        Returns:
            品質報告
        """
        logger.info("檢查資料品質...")
        
        report = {
            'total_rows': len(df),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max()),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated(subset=['date']).sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # 檢查資料連續性
        if len(df) > 1:
            date_diff = df['date'].diff().dt.days
            gaps = date_diff[date_diff > 3]  # 超過 3 天視為間隔
            
            report['continuity'] = {
                'total_gaps': len(gaps),
                'max_gap_days': int(date_diff.max()) if not date_diff.isna().all() else 0
            }
        
        # 統計摘要
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report['statistics'] = df[numeric_cols].describe().to_dict()
        
        logger.info(f"[OK] 資料品質檢查完成")
        logger.info(f"   資料筆數: {report['total_rows']}")
        logger.info(f"   日期範圍: {report['date_range']['start']} ~ {report['date_range']['end']}")
        logger.info(f"   資料天數: {report['date_range']['days']}")
        
        return report


# 測試程式碼
if __name__ == "__main__":
    from src.data.database import Database
    
    logger.info("=== 測試資料清洗模組 ===")
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料")
        exit(1)
    
    # 清洗資料
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_futures_data(df)
    
    # 檢查品質
    quality_report = cleaner.check_data_quality(cleaned_df)
    
    print("\n" + "="*60)
    print("資料品質報告")
    print("="*60)
    print(f"總筆數: {quality_report['total_rows']}")
    print(f"日期範圍: {quality_report['date_range']['start']} ~ {quality_report['date_range']['end']}")
    print(f"資料天數: {quality_report['date_range']['days']}")
    print(f"重複資料: {quality_report['duplicates']}")
    
    if quality_report.get('continuity'):
        print(f"資料間隔: {quality_report['continuity']['total_gaps']} 個")
        print(f"最大間隔: {quality_report['continuity']['max_gap_days']} 天")
    
    print("="*60)
