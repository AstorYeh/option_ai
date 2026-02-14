"""
系統設定頁面
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import os
from dotenv import load_dotenv, set_key
from src.data.finmind_client import FinMindClient
from src.models.llm_advisor import LLMAdvisor
from src.notification.discord_bot import DiscordNotifier

st.set_page_config(page_title="系統設定", page_icon="⚙️", layout="wide")

st.title("⚙️ 系統設定")
st.markdown("---")

# 載入環境變數
load_dotenv()
env_file = Path(__file__).parent.parent.parent / ".env"

# API 設定
st.header("🔑 API 設定")

col1, col2 = st.columns(2)

with col1:
    st.subheader("FinMind API")
    finmind_token = st.text_input(
        "API Token",
        value=os.getenv("FINMIND_API_TOKEN", ""),
        type="password",
        help="在 https://finmindtrade.com/ 註冊取得"
    )
    
    if st.button("測試 FinMind 連線"):
        if finmind_token:
            os.environ["FINMIND_API_TOKEN"] = finmind_token
            client = FinMindClient(finmind_token)
            if client.test_connection():
                st.success("✅ FinMind API 連線成功!")
                # 儲存到 .env
                if env_file.exists():
                    set_key(str(env_file), "FINMIND_API_TOKEN", finmind_token)
            else:
                st.error("❌ FinMind API 連線失敗")
        else:
            st.warning("請輸入 API Token")

with col2:
    st.subheader("Ollama LLM")
    ollama_url = st.text_input(
        "API URL",
        value=os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    )
    ollama_model = st.text_input(
        "模型名稱",
        value=os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    )
    
    if st.button("測試 Ollama 連線"):
        os.environ["OLLAMA_API_URL"] = ollama_url
        os.environ["OLLAMA_MODEL"] = ollama_model
        advisor = LLMAdvisor(ollama_url, ollama_model)
        if advisor.test_connection():
            st.success("✅ Ollama 連線成功!")
            # 儲存到 .env
            if env_file.exists():
                set_key(str(env_file), "OLLAMA_API_URL", ollama_url)
                set_key(str(env_file), "OLLAMA_MODEL", ollama_model)
        else:
            st.error("❌ Ollama 連線失敗")

st.markdown("---")

# Discord 設定
st.header("📢 Discord 通知設定")

discord_webhook = st.text_input(
    "Webhook URL",
    value=os.getenv("DISCORD_WEBHOOK_URL", ""),
    type="password",
    help="在 Discord 伺服器設定 > 整合 > Webhook 取得"
)

col1, col2, col3 = st.columns(3)

with col1:
    enable_notify = st.checkbox(
        "啟用通知",
        value=os.getenv("ENABLE_DISCORD_NOTIFY", "true").lower() == "true"
    )

with col2:
    notify_on_signal = st.checkbox(
        "訊號通知",
        value=os.getenv("NOTIFY_ON_SIGNAL", "true").lower() == "true"
    )

with col3:
    notify_on_error = st.checkbox(
        "錯誤通知",
        value=os.getenv("NOTIFY_ON_ERROR", "true").lower() == "true"
    )

if st.button("測試 Discord 通知"):
    if discord_webhook:
        os.environ["DISCORD_WEBHOOK_URL"] = discord_webhook
        notifier = DiscordNotifier(discord_webhook)
        if notifier.test_connection():
            st.success("✅ Discord 通知測試成功!")
            # 儲存到 .env
            if env_file.exists():
                set_key(str(env_file), "DISCORD_WEBHOOK_URL", discord_webhook)
                set_key(str(env_file), "ENABLE_DISCORD_NOTIFY", str(enable_notify).lower())
                set_key(str(env_file), "NOTIFY_ON_SIGNAL", str(notify_on_signal).lower())
                set_key(str(env_file), "NOTIFY_ON_ERROR", str(notify_on_error).lower())
        else:
            st.error("❌ Discord 通知測試失敗")
    else:
        st.warning("請輸入 Webhook URL")

st.markdown("---")

# 風險管理設定
st.header("⚠️ 風險管理設定")

col1, col2 = st.columns(2)

with col1:
    max_position = st.number_input(
        "最大部位數",
        min_value=1,
        max_value=10,
        value=int(os.getenv("MAX_POSITION_SIZE", "2")),
        help="同時持有的最大選擇權口數"
    )
    
    stop_loss = st.number_input(
        "停損百分比 (%)",
        min_value=10,
        max_value=100,
        value=int(os.getenv("STOP_LOSS_PERCENT", "50")),
        help="權利金跌破此百分比時停損"
    )

with col2:
    take_profit = st.number_input(
        "停利百分比 (%)",
        min_value=50,
        max_value=500,
        value=int(os.getenv("TAKE_PROFIT_PERCENT", "100")),
        help="權利金達到此百分比時停利"
    )

if st.button("儲存風險設定"):
    if env_file.exists():
        set_key(str(env_file), "MAX_POSITION_SIZE", str(max_position))
        set_key(str(env_file), "STOP_LOSS_PERCENT", str(stop_loss))
        set_key(str(env_file), "TAKE_PROFIT_PERCENT", str(take_profit))
        st.success("✅ 風險設定已儲存!")

st.markdown("---")

# 系統資訊
st.header("ℹ️ 系統資訊")

col1, col2 = st.columns(2)

with col1:
    st.subheader("環境變數檔案")
    if env_file.exists():
        st.success(f"✅ {env_file}")
    else:
        st.error(f"❌ 找不到 .env 檔案")
        st.info("請複製 .env.example 為 .env")

with col2:
    st.subheader("資料庫")
    db_path = Path(__file__).parent.parent.parent / "data" / "database" / "options.db"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        st.success(f"✅ 資料庫大小: {size_mb:.2f} MB")
    else:
        st.warning("⚠️ 資料庫尚未建立")
        st.info("執行: python scripts/init_database.py")
