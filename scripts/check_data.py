"""
檢查資料庫狀態
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
import pandas as pd

with Database() as db:
    df = db.get_futures_data()
    print(f"資料庫總筆數: {len(df)}")
    print(f"日期範圍: {df['date'].min()} ~ {df['date'].max()}")
    print(f"\n最近10筆:")
    print(df.tail(10)[['date', 'open', 'close', 'volume']])
