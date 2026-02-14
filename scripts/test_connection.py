"""
測試系統連線
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.finmind_client import FinMindClient
from src.models.llm_advisor import LLMAdvisor
from src.notification.discord_bot import DiscordNotifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """測試所有系統連線"""
    logger.info("=== 開始系統連線測試 ===\n")
    
    all_passed = True
    
    # 測試 FinMind API
    logger.info("1. 測試 FinMind API...")
    try:
        client = FinMindClient()
        if client.test_connection():
            logger.info("[OK] FinMind API 連線成功\n")
        else:
            logger.error("[ERROR] FinMind API 連線失敗\n")
            all_passed = False
    except Exception as e:
        logger.error(f"[ERROR] FinMind API 測試失敗: {e}\n")
        all_passed = False
    
    # 測試 Ollama LLM
    logger.info("2. 測試 Ollama LLM...")
    try:
        advisor = LLMAdvisor()
        if advisor.test_connection():
            logger.info("[OK] Ollama LLM 連線成功\n")
        else:
            logger.error("[ERROR] Ollama LLM 連線失敗\n")
            logger.info("提示: 請確認 Ollama 服務是否運行中")
            logger.info("啟動命令: ollama serve\n")
            all_passed = False
    except Exception as e:
        logger.error(f"[ERROR] Ollama LLM 測試失敗: {e}\n")
        all_passed = False
    
    # 測試 Discord Webhook
    logger.info("3. 測試 Discord Webhook...")
    try:
        notifier = DiscordNotifier()
        if notifier.test_connection():
            logger.info("[OK] Discord Webhook 連線成功\n")
        else:
            logger.warning("[WARN] Discord Webhook 未設定或連線失敗\n")
    except Exception as e:
        logger.error(f"[ERROR] Discord Webhook 測試失敗: {e}\n")
    
    # 總結
    logger.info("=== 測試完成 ===")
    if all_passed:
        logger.info("[OK] 所有核心服務連線正常!")
    else:
        logger.warning("[WARN] 部分服務連線失敗,請檢查設定")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
