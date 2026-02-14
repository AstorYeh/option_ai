"""
全域設定管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 專案根目錄
BASE_DIR = Path(__file__).parent.parent

# 資料目錄
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
DATABASE_DIR = DATA_DIR / "database"

# 模型目錄
MODEL_DIR = BASE_DIR / "models"

# 日誌目錄
LOG_DIR = BASE_DIR / "logs"

# 確保目錄存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, 
                  DATABASE_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 資料庫設定
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATABASE_DIR / "options.db"))

# 日誌設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 時區設定
TIMEZONE = "Asia/Taipei"

# 台指期設定
FUTURES_SYMBOL = "TX"  # 台指期代碼
FUTURES_EXCHANGE = "TAIFEX"  # 台灣期貨交易所

# 選擇權設定
OPTIONS_SYMBOL = "TXO"  # 台指選擇權代碼

# 交易時間設定
MARKET_OPEN_TIME = "08:45"
MARKET_CLOSE_TIME = "13:45"
AFTER_MARKET_CLOSE_TIME = "15:00"

# 資料更新設定
DATA_UPDATE_TIME = os.getenv("DATA_UPDATE_TIME", "15:30")
PREDICTION_TIME = os.getenv("PREDICTION_TIME", "16:00")

# 回測設定
BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2022-01-01")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2024-12-31")
INITIAL_CAPITAL = int(os.getenv("INITIAL_CAPITAL", "1000000"))

# 風險管理設定
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "2"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "50"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "100"))

# 模型設定
RETRAIN_INTERVAL_DAYS = int(os.getenv("RETRAIN_INTERVAL_DAYS", "7"))

# 通知設定
ENABLE_DISCORD_NOTIFY = os.getenv("ENABLE_DISCORD_NOTIFY", "true").lower() == "true"
NOTIFY_ON_SIGNAL = os.getenv("NOTIFY_ON_SIGNAL", "true").lower() == "true"
NOTIFY_ON_ERROR = os.getenv("NOTIFY_ON_ERROR", "true").lower() == "true"
DAILY_REPORT_TIME = os.getenv("DAILY_REPORT_TIME", "17:00")
