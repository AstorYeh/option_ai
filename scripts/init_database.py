"""
資料庫初始化腳本
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """初始化資料庫"""
    logger.info("開始初始化資料庫...")
    
    try:
        with Database() as db:
            db.create_tables()
        
        logger.info("[OK] 資料庫初始化完成!")
        
    except Exception as e:
        logger.error(f"[ERROR] 資料庫初始化失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
