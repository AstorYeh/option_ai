"""
Step 1: 資料準備與分割 (80/20)
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.data.database import Database
from src.features.technical import add_all_technical_indicators
from src.features.external import add_external_features

def prepare_data():
    """準備訓練與測試資料"""
    logger.info("=== 開始資料準備 ===")
    
    # 建立輸出目錄
    data_dir = Path("data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    features_dir = data_dir / "features"
    
    for dir_path in [train_dir, test_dir, features_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 載入原始資料
    logger.info("載入原始資料...")
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty:
        logger.error("無資料!")
        return False
    
    logger.info(f"原始資料: {len(df)} 筆")
    logger.info(f"日期範圍: {df['date'].min()} ~ {df['date'].max()}")
    
    # 計算技術指標
    logger.info("計算技術指標...")
    df = add_all_technical_indicators(df)
    
    # 加入外部因子 (美股/權值股/法人留倉)
    logger.info("加入外部因子...")
    try:
        df = add_external_features(df)
    except Exception as e:
        logger.warning(f"外部因子載入失敗 (將繼續): {e}")
    
    # 填補缺失值而非移除
    logger.info("處理缺失值...")
    
    # 數值欄位用前向填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    
    # 處理無限值
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    logger.info(f"處理後資料: {len(df)} 筆")
    
    # 移除明顯異常的資料(價格為0或成交量為0)
    before = len(df)
    df = df[(df['close'] > 0) & (df['volume'] > 0)]
    removed = before - len(df)
    if removed > 0:
        logger.info(f"移除異常資料: {removed} 筆")
    
    logger.info(f"可用資料: {len(df)} 筆")
    
    # 按時間排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 80/20 分割
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"\n=== 資料分割 ===")
    logger.info(f"訓練集: {len(train_df)} 筆 ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  日期範圍: {train_df['date'].min()} ~ {train_df['date'].max()}")
    logger.info(f"測試集: {len(test_df)} 筆 ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"  日期範圍: {test_df['date'].min()} ~ {test_df['date'].max()}")
    
    # 儲存資料
    logger.info("\n儲存資料...")
    
    # 儲存完整資料 (含所有欄位)
    train_df.to_csv(train_dir / "train_full.csv", index=False)
    test_df.to_csv(test_dir / "test_full.csv", index=False)
    logger.info(f"[OK] 完整資料已儲存")
    
    # 特徵資料 (排除原始價格, 防止價格洩漏)
    # 原始 OHLCV 和 spread 為絕對價格, 不具備跨時段預測能力
    raw_price_cols = ['open', 'high', 'low', 'close', 'volume',
                      'spread', 'spread_per', 'atr']
    exclude_cols = ['date', 'futures_id', 'contract_date', 'trading_session'] + raw_price_cols
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols
                    and not col.startswith('return_')
                    and not col.startswith('label_')]
    
    train_features = train_df[feature_cols]
    test_features = test_df[feature_cols]
    
    train_features.to_csv(features_dir / "train_features.csv", index=False)
    test_features.to_csv(features_dir / "test_features.csv", index=False)
    logger.info(f"[OK] 特徵資料已儲存 ({len(feature_cols)} 個特徵)")
    
    # 建立標籤 (未來 N 天漲跌)
    logger.info("\n建立預測標籤...")
    
    for horizon in [1, 3, 5]:
        # 計算未來 N 天報酬率
        train_df[f'return_{horizon}d'] = train_df['close'].pct_change(horizon).shift(-horizon)
        test_df[f'return_{horizon}d'] = test_df['close'].pct_change(horizon).shift(-horizon)
        
        # 三分類標籤: 1=漲, 0=盤, -1=跌
        threshold = 0.01  # 1% 閾值
        train_df[f'label_{horizon}d'] = np.where(
            train_df[f'return_{horizon}d'] > threshold, 1,
            np.where(train_df[f'return_{horizon}d'] < -threshold, -1, 0)
        )
        test_df[f'label_{horizon}d'] = np.where(
            test_df[f'return_{horizon}d'] > threshold, 1,
            np.where(test_df[f'return_{horizon}d'] < -threshold, -1, 0)
        )
        
        logger.info(f"  {horizon}天標籤分布:")
        logger.info(f"    訓練集 - 漲:{(train_df[f'label_{horizon}d']==1).sum()}, "
                   f"盤:{(train_df[f'label_{horizon}d']==0).sum()}, "
                   f"跌:{(train_df[f'label_{horizon}d']==-1).sum()}")
    
    # 儲存含標籤的資料
    train_df.to_csv(train_dir / "train_labeled.csv", index=False)
    test_df.to_csv(test_dir / "test_labeled.csv", index=False)
    logger.info(f"[OK] 標籤資料已儲存")
    
    # 生成資料摘要
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_date_range': f"{train_df['date'].min()} ~ {train_df['date'].max()}",
        'test_date_range': f"{test_df['date'].min()} ~ {test_df['date'].max()}",
        'num_features': len(feature_cols),
        'feature_names': feature_cols
    }
    
    import json
    with open(data_dir / "data_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("\n=== 資料準備完成 ===")
    logger.info(f"輸出目錄: {data_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    success = prepare_data()
    
    if success:
        print("\n" + "="*60)
        print("[SUCCESS] 資料準備完成!")
        print("="*60)
        print("\n下一步: python scripts/2_train_models.py")
    else:
        print("\n" + "="*60)
        print("[ERROR] 資料準備失敗!")
        print("="*60)
