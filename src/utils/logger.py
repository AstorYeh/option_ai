"""
日誌系統配置
"""
from loguru import logger
import sys
from pathlib import Path
from config.settings import LOG_DIR, LOG_LEVEL

# 移除預設 handler
logger.remove()

# 控制台輸出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True
)

# 應用程式主日誌
logger.add(
    LOG_DIR / "app.log",
    rotation="1 day",
    retention="30 days",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    encoding="utf-8"
)

# 資料抓取日誌
logger.add(
    LOG_DIR / "data_fetch.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    filter=lambda record: "data" in record["name"],
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    encoding="utf-8"
)

# 預測日誌
logger.add(
    LOG_DIR / "prediction.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    filter=lambda record: "models" in record["name"],
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    encoding="utf-8"
)

# 回測日誌
logger.add(
    LOG_DIR / "backtest.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    filter=lambda record: "backtest" in record["name"],
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    encoding="utf-8"
)

# 錯誤日誌
logger.add(
    LOG_DIR / "error.log",
    rotation="1 week",
    retention="60 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    encoding="utf-8"
)

def get_logger(name: str):
    """取得指定名稱的 logger"""
    return logger.bind(name=name)
