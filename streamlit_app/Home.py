"""
Streamlit 主頁
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
from src.data.database import Database
from src.utils.helpers import get_taiwan_time, format_currency, format_percentage

# 頁面配置
st.set_page_config(
    page_title="台指期選擇權預測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 標題
st.title("📊 台指期選擇權買方策略預測系統")
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.header("⚙️ 系統資訊")
    st.info(f"⏰ 當前時間\n\n{get_taiwan_time().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    st.header("📌 快速導航")
    st.page_link("pages/1_📊_即時預測.py", label="📊 即時預測", icon="📊")
    st.page_link("pages/2_📈_回測分析.py", label="📈 回測分析", icon="📈")
    st.page_link("pages/3_📉_績效追蹤.py", label="📉 績效追蹤", icon="📉")
    st.page_link("pages/4_⚙️_系統設定.py", label="⚙️ 系統設定", icon="⚙️")

# 主要內容
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🎯 系統狀態",
        value="運行中",
        delta="正常"
    )

with col2:
    st.metric(
        label="📊 資料更新",
        value="今日",
        delta="最新"
    )

with col3:
    st.metric(
        label="🤖 AI 模型",
        value="已載入",
        delta="就緒"
    )

st.markdown("---")

# 系統概覽
st.header("📋 系統概覽")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 核心功能")
    st.markdown("""
    - ✅ **即時預測**: AI 分析當前市場,提供 Buy Call/Put 建議
    - ✅ **波動率分析**: IV/HV 比值,識別低波動進場時機
    - ✅ **LLM 策略顧問**: Local LLM 提供專業交易建議
    - ✅ **回測驗證**: 歷史資料回測,驗證策略有效性
    - ✅ **Discord 通知**: 即時推播進場訊號
    """)

with col2:
    st.subheader("⚠️ 風險聲明")
    st.warning("""
    **本系統僅供學習與研究用途,不構成投資建議。**
    
    - 選擇權交易具有高風險
    - 可能導致全部權利金損失
    - 歷史績效不代表未來表現
    - 請做好風險管理
    """)

st.markdown("---")

# 最新市場數據
st.header("📊 最新市場數據")

try:
    with Database() as db:
        # 取得最新台指期資料
        latest_futures = db.get_futures_data()
        
        if not latest_futures.empty:
            latest = latest_futures.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="收盤價",
                    value=f"{latest['close']:.0f}",
                    delta=f"{latest['close'] - latest['open']:.0f}"
                )
            
            with col2:
                st.metric(
                    label="最高價",
                    value=f"{latest['high']:.0f}"
                )
            
            with col3:
                st.metric(
                    label="最低價",
                    value=f"{latest['low']:.0f}"
                )
            
            with col4:
                st.metric(
                    label="成交量",
                    value=f"{latest['volume']:,.0f}"
                )
            
            # 顯示最近 10 天資料
            st.subheader("📈 最近 10 天走勢")
            recent_data = latest_futures.tail(10)[['date', 'open', 'high', 'low', 'close', 'volume']]
            st.dataframe(recent_data, width='stretch')
            
            # 簡單圖表
            st.line_chart(latest_futures.tail(30).set_index('date')['close'])
        else:
            st.warning("⚠️ 尚無市場資料,請先執行資料更新")
            st.code("python scripts/daily_update.py --initial", language="bash")

except Exception as e:
    st.error(f"❌ 載入資料失敗: {e}")

st.markdown("---")

# 快速開始指南
st.header("🚀 快速開始")

tab1, tab2, tab3 = st.tabs(["📥 初始化", "📊 使用流程", "⚙️ 設定"])

with tab1:
    st.markdown("""
    ### 首次使用設定
    
    1. **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```
    
    2. **設定環境變數**
    - 複製 `.env.example` 為 `.env`
    - 填入 FinMind API Token
    - 設定 Discord Webhook URL
    
    3. **初始化資料庫**
    ```bash
    python scripts/init_database.py
    ```
    
    4. **下載歷史資料**
    ```bash
    python scripts/daily_update.py --initial
    ```
    
    5. **啟動系統**
    ```bash
    streamlit run streamlit_app/Home.py
    ```
    """)

with tab2:
    st.markdown("""
    ### 日常使用流程
    
    1. **每日資料更新** (收盤後執行)
    ```bash
    python scripts/daily_update.py
    ```
    
    2. **查看即時預測**
    - 前往「📊 即時預測」頁面
    - 查看 AI 預測結果與 LLM 建議
    
    3. **回測策略**
    - 前往「📈 回測分析」頁面
    - 選擇日期範圍執行回測
    
    4. **追蹤績效**
    - 前往「📉 績效追蹤」頁面
    - 記錄實際交易並追蹤績效
    """)

with tab3:
    st.markdown("""
    ### 系統設定
    
    前往「⚙️ 系統設定」頁面可調整:
    - API 金鑰
    - Discord 通知規則
    - 模型參數
    - 風險管理設定
    """)

st.markdown("---")

# 頁尾
st.caption("© 2026 台指期選擇權預測系統 | 僅供學習研究使用")
