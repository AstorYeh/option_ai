"""
API 配置管理
"""
import os
from dotenv import load_dotenv

load_dotenv()

# FinMind API 設定
FINMIND_API_TOKEN = os.getenv("FINMIND_API_TOKEN", "")
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# Ollama API 設定
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Discord Webhook 設定
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# API 請求設定
REQUEST_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# FinMind API 限制 (免費版)
# 免費版: 300 次/小時
# 註冊會員: 600 次/小時
FINMIND_RATE_LIMIT = 600  # 每小時請求次數(註冊會員)
FINMIND_REQUEST_DELAY = 1.0  # 請求間隔(秒) - 避免超過限制
