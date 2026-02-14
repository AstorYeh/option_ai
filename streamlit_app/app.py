"""
台指期選擇權買方策略預測系統 - Streamlit 主程式
"""
import streamlit as st
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 頁面配置
st.set_page_config(
    page_title="台指期選擇權預測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00D9FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #00D9FF;
    }
</style>
""", unsafe_allow_html=True)

# 主標題
st.markdown('<div class="main-header">📊 台指期選擇權買方策略預測系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI 驅動的選擇權交易決策支援系統</div>', unsafe_allow_html=True)

# 系統簡介
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🤖 AI 預測引擎
    
    - **XGBoost** 方向性預測
    - **LSTM** 波動率預測
    - **LLM** 策略建議
    - **集成系統** 綜合判斷
    """)

with col2:
    st.markdown("""
    ### 📈 回測驗證
    
    - 歷史資料回測
    - 多種績效指標
    - 參數自動優化
    - 風險評估報告
    """)

with col3:
    st.markdown("""
    ### 🌐 Web 介面
    
    - 即時預測展示
    - 互動式圖表
    - 績效追蹤
    - 參數設定
    """)

# 快速開始
st.markdown("---")
st.header("🚀 快速開始")

st.markdown("""
### 使用步驟

1. **📊 即時預測**: 查看最新的 AI 預測結果與 LLM 建議
2. **📈 回測分析**: 驗證策略的歷史績效表現
3. **📉 績效追蹤**: 監控交易表現與市場狀況
4. **⚙️ 系統設定**: 調整模型參數與通知設定

請從左側選單選擇功能頁面開始使用!
""")

# 系統狀態
st.markdown("---")
st.header("📊 系統狀態")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="資料筆數",
        value="1,041",
        delta="90 天"
    )

with col2:
    st.metric(
        label="技術指標",
        value="22 個",
        delta="完整"
    )

with col3:
    st.metric(
        label="AI 模型",
        value="3 個",
        delta="已訓練"
    )

with col4:
    st.metric(
        label="系統狀態",
        value="運行中",
        delta="正常"
    )

# 功能特色
st.markdown("---")
st.header("✨ 功能特色")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🎯 核心功能
    
    - ✅ **方向性預測**: XGBoost 三分類模型 (漲/跌/盤整)
    - ✅ **波動率預測**: 預測未來價格波動
    - ✅ **LLM 建議**: Ollama 本地 LLM 策略分析
    - ✅ **集成預測**: 整合多模型結果
    - ✅ **回測系統**: 完整的歷史績效驗證
    - ✅ **參數優化**: 自動尋找最佳參數組合
    """)

with col2:
    st.markdown("""
    #### 📊 技術指標
    
    - RSI, MACD, 布林通道
    - ATR, ADX, 動量指標
    - 移動平均線 (SMA, EMA)
    - 歷史波動率
    - Greeks 計算
    - IV/HV 比值
    """)

# 免責聲明
st.markdown("---")
st.warning("""
⚠️ **免責聲明**

本系統僅供學習與研究使用,預測結果不構成投資建議。
實際交易請自行評估風險,選擇權交易具有高風險,請謹慎操作。
""")

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p>台指期選擇權買方策略預測系統 v1.0.0</p>
    <p>© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
