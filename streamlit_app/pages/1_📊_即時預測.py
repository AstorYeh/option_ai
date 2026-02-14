"""
即時預測頁面
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.database import Database
from src.data.finmind_client import FinMindClient
from src.features.technical import add_all_technical_indicators
from src.features.options_metrics import (
    calculate_historical_volatility,
    analyze_options_chain
)
from src.features.greeks import BlackScholesGreeks
from src.models.llm_advisor import LLMAdvisor
from src.notification.discord_bot import DiscordNotifier
from src.utils.helpers import get_strike_prices, format_percentage

st.set_page_config(page_title="即時預測", page_icon="📊", layout="wide")

st.title("📊 即時預測")
st.markdown("---")

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 預測設定")
    
    use_llm = st.checkbox("啟用 LLM 建議", value=True)
    send_discord = st.checkbox("發送 Discord 通知", value=False)
    
    st.markdown("---")
    
    confidence_threshold = st.slider(
        "信心度閾值",
        min_value=0.5,
        max_value=0.9,
        value=0.65,
        step=0.05,
        help="低於此閾值將建議觀望"
    )

# 主要內容
try:
    with Database() as db:
        # 取得最新資料
        futures_df = db.get_futures_data()
        
        if futures_df.empty:
            st.warning("⚠️ 尚無資料,請先執行資料更新")
            st.stop()
        
        # 計算技術指標
        with st.spinner("計算技術指標..."):
            futures_with_indicators = add_all_technical_indicators(futures_df)
        
        latest = futures_with_indicators.iloc[-1]
        
        # 顯示當前市場狀況
        st.header("📊 當前市場狀況")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            change = latest['close'] - latest['open']
            change_pct = (change / latest['open']) * 100
            st.metric(
                "台指期收盤",
                f"{latest['close']:.0f}",
                f"{change:+.0f} ({change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("RSI", f"{latest['rsi']:.1f}")
        
        with col3:
            st.metric("MACD", f"{latest['macd']:.1f}")
        
        with col4:
            hv = calculate_historical_volatility(futures_with_indicators)
            st.metric("歷史波動率", f"{hv.iloc[-1]:.2%}")
        
        st.markdown("---")
        
        # 技術指標詳情
        with st.expander("📈 技術指標詳情"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("趨勢指標")
                st.write(f"MA5: {latest.get('MA5', 0):.0f}")
                st.write(f"MA20: {latest.get('MA20', 0):.0f}")
                st.write(f"MA60: {latest.get('MA60', 0):.0f}")
            
            with col2:
                st.subheader("動量指標")
                st.write(f"RSI: {latest['rsi']:.1f}")
                st.write(f"MACD: {latest['macd']:.1f}")
                st.write(f"ATR: {latest['atr']:.1f}")
            
            with col3:
                st.subheader("波動率")
                st.write(f"布林通道寬度: {latest.get('bb_width', 0):.3f}")
                st.write(f"歷史波動率: {hv.iloc[-1]:.2%}")
        
        st.markdown("---")
        
        # 簡化的預測邏輯(示範用)
        st.header("🤖 AI 預測結果")
        
        # 基於技術指標的簡單預測
        prediction = {
            'direction': 'neutral',
            'confidence': 0.5,
            'predicted_change': 0.0
        }
        
        # RSI 判斷
        if latest['rsi'] > 70:
            prediction['direction'] = 'bearish'
            prediction['confidence'] = min(0.8, (latest['rsi'] - 70) / 30 + 0.5)
        elif latest['rsi'] < 30:
            prediction['direction'] = 'bullish'
            prediction['confidence'] = min(0.8, (30 - latest['rsi']) / 30 + 0.5)
        
        # MACD 輔助判斷
        if latest['macd'] > 0 and latest['macd_histogram'] > 0:
            if prediction['direction'] == 'bullish':
                prediction['confidence'] = min(0.9, prediction['confidence'] + 0.1)
            else:
                prediction['direction'] = 'bullish'
                prediction['confidence'] = 0.6
        elif latest['macd'] < 0 and latest['macd_histogram'] < 0:
            if prediction['direction'] == 'bearish':
                prediction['confidence'] = min(0.9, prediction['confidence'] + 0.1)
            else:
                prediction['direction'] = 'bearish'
                prediction['confidence'] = 0.6
        
        # 預測漲跌幅
        if prediction['direction'] == 'bullish':
            prediction['predicted_change'] = 1.0 * prediction['confidence']
        elif prediction['direction'] == 'bearish':
            prediction['predicted_change'] = -1.0 * prediction['confidence']
        
        # 顯示預測結果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_emoji = "🚀" if prediction['direction'] == 'bullish' else "📉" if prediction['direction'] == 'bearish' else "⏸️"
            direction_text = "看漲" if prediction['direction'] == 'bullish' else "看跌" if prediction['direction'] == 'bearish' else "中性"
            st.metric("方向預測", f"{direction_emoji} {direction_text}")
        
        with col2:
            st.metric("信心度", f"{prediction['confidence']:.1%}")
        
        with col3:
            st.metric("預測漲跌幅", f"{prediction['predicted_change']:+.2f}%")
        
        # 模擬選擇權分析
        options_analysis = {
            'pcr_volume': 0.9,
            'avg_iv': hv.iloc[-1] * 1.1,  # 假設 IV 略高於 HV
            'iv_hv_ratio': 1.1,
            'volatility_environment': 'normal',
            'sentiment': 'neutral',
            'max_pain': latest['close']
        }
        
        st.markdown("---")
        
        # LLM 建議
        if use_llm:
            st.header("💡 LLM 策略建議")
            
            with st.spinner("正在請求 LLM 建議..."):
                try:
                    advisor = LLMAdvisor()
                    
                    market_data = {
                        'close': latest['close'],
                        'change': change,
                        'change_pct': change_pct,
                        'volume': latest['volume'],
                        'rsi': latest['rsi'],
                        'macd': latest['macd'],
                        'bb_position': 'middle',
                        'atr': latest['atr'],
                        'hv': hv.iloc[-1]
                    }
                    
                    advice = advisor.get_trading_advice(market_data, prediction, options_analysis)
                    
                    # 顯示建議
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        action_color = "green" if advice['action'] == 'BUY_CALL' else "red" if advice['action'] == 'BUY_PUT' else "gray"
                        st.markdown(f"### :{action_color}[{advice['action']}]")
                    
                    with col2:
                        if advice['strike_price']:
                            st.metric("建議履約價", f"{advice['strike_price']}")
                        else:
                            st.metric("建議履約價", "N/A")
                    
                    with col3:
                        risk_color = "green" if advice['risk_level'] == 'low' else "orange" if advice['risk_level'] == 'medium' else "red"
                        st.markdown(f"### 風險: :{risk_color}[{advice['risk_level'].upper()}]")
                    
                    st.info(f"**進場理由**: {advice.get('reasoning', '無')}")
                    st.warning(f"**停損停利**: {advice.get('stop_loss', '無')}")
                    
                    if advice.get('warnings'):
                        st.error(f"**注意事項**: {advice['warnings']}")
                    
                    # Discord 通知
                    if send_discord and advice['action'] != 'HOLD':
                        if st.button("📤 發送 Discord 通知"):
                            notifier = DiscordNotifier()
                            notifier.send_signal(
                                advice['action'],
                                market_data,
                                prediction,
                                advice,
                                options_analysis
                            )
                            st.success("✅ Discord 通知已發送!")
                    
                except Exception as e:
                    st.error(f"❌ LLM 請求失敗: {e}")
                    st.info("請確認 Ollama 服務是否運行中")
        
        st.markdown("---")
        
        # 履約價選擇工具
        st.header("🎯 履約價選擇工具")
        
        strikes = get_strike_prices(latest['close'], num_strikes=5)
        
        strike_data = []
        for strike in strikes:
            bs = BlackScholesGreeks(
                spot_price=latest['close'],
                strike_price=strike,
                time_to_expiry=30/365,
                volatility=hv.iloc[-1]
            )
            
            call_greeks = bs.get_all_greeks('call')
            put_greeks = bs.get_all_greeks('put')
            
            moneyness = "價平" if abs(latest['close'] - strike) < 100 else \
                       "價內" if latest['close'] > strike else "價外"
            
            strike_data.append({
                '履約價': strike,
                '價內外': moneyness,
                'Call 價格': f"{call_greeks['price']:.0f}",
                'Call Delta': f"{call_greeks['delta']:.2f}",
                'Put 價格': f"{put_greeks['price']:.0f}",
                'Put Delta': f"{put_greeks['delta']:.2f}",
                'Gamma': f"{call_greeks['gamma']:.4f}",
                'Theta': f"{call_greeks['theta']:.2f}",
                'Vega': f"{call_greeks['vega']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(strike_data), use_container_width=True)

except Exception as e:
    st.error(f"❌ 發生錯誤: {e}")
    import traceback
    st.code(traceback.format_exc())
