"""
績效追蹤頁面
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import Database

st.set_page_config(page_title="績效追蹤", page_icon="📉", layout="wide")

st.title("📉 績效追蹤")
st.markdown("---")

# 載入資料
@st.cache_data(ttl=300)
def load_data():
    with Database() as db:
        # 載入台指期資料
        futures_df = db.get_futures_data()
        
        # 載入交易記錄 (如果有)
        try:
            trades_df = db.get_trade_history(limit=100)
        except:
            trades_df = pd.DataFrame()
        
        # 載入預測記錄 (如果有)
        try:
            predictions_df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY date DESC LIMIT 100",
                db.conn
            )
        except:
            predictions_df = pd.DataFrame()
    
    return futures_df, trades_df, predictions_df

try:
    futures_df, trades_df, predictions_df = load_data()
except Exception as e:
    st.error(f"載入資料失敗: {e}")
    st.stop()

# 側邊欄篩選
st.sidebar.header("篩選條件")

# 日期範圍
date_range = st.sidebar.date_input(
    "日期範圍",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# 市場總覽
st.header("📊 市場總覽")

if not futures_df.empty:
    latest = futures_df.iloc[-1]
    prev = futures_df.iloc[-2] if len(futures_df) > 1 else latest
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        change = latest['close'] - prev['close']
        change_pct = (change / prev['close']) * 100 if prev['close'] > 0 else 0
        st.metric(
            "台指期收盤價",
            f"{latest['close']:,.0f}",
            delta=f"{change:+.0f} ({change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "成交量",
            f"{latest['volume']:,.0f}",
            delta=f"{latest['volume'] - prev['volume']:+,.0f}"
        )
    
    with col3:
        high_low_range = latest['high'] - latest['low']
        st.metric(
            "當日波動",
            f"{high_low_range:.0f}",
            delta="點"
        )
    
    with col4:
        # 計算近期平均波動
        recent_volatility = futures_df.tail(20)['close'].pct_change().std() * 100
        st.metric(
            "近20日波動率",
            f"{recent_volatility:.2f}%",
            delta="標準差"
        )
    
    # 價格走勢圖
    st.markdown("---")
    st.subheader("📈 價格走勢")
    
    # 篩選日期範圍
    if len(date_range) == 2:
        mask = (futures_df['date'] >= pd.Timestamp(date_range[0])) & \
               (futures_df['date'] <= pd.Timestamp(date_range[1]))
        filtered_df = futures_df[mask]
    else:
        filtered_df = futures_df.tail(30)
    
    # K線圖 (台灣習慣: 漲紅跌綠)
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='台指期',
        increasing_line_color='red',  # 上漲為紅色
        decreasing_line_color='green'  # 下跌為綠色
    )])
    
    fig_candlestick.update_layout(
        title="台指期 K 線圖 (漲紅跌綠)",
        xaxis_title="日期",
        yaxis_title="價格",
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig_candlestick, width='stretch')
    
    # 成交量圖
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=filtered_df['date'],
        y=filtered_df['volume'],
        name='成交量',
        marker_color='#00D9FF'
    ))
    
    fig_volume.update_layout(
        title="成交量",
        xaxis_title="日期",
        yaxis_title="成交量",
        height=300
    )
    
    st.plotly_chart(fig_volume, width='stretch')

# 交易績效
st.markdown("---")
st.header("💼 交易績效")

if not trades_df.empty:
    # 績效指標
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
    losing_trades = len(trades_df[trades_df['profit_loss'] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_pnl = trades_df['profit_loss'].sum()
    avg_pnl = trades_df['profit_loss'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總交易次數", f"{total_trades}")
    
    with col2:
        st.metric("勝率", f"{win_rate:.2%}")
    
    with col3:
        st.metric("總損益", f"${total_pnl:+,.0f}")
    
    with col4:
        st.metric("平均損益", f"${avg_pnl:+,.0f}")
    
    # 累積損益曲線
    trades_df['cumulative_pnl'] = trades_df['profit_loss'].cumsum()
    
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=trades_df['entry_date'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='累積損益',
        line=dict(color='#00D9FF', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    fig_pnl.update_layout(
        title="累積損益曲線",
        xaxis_title="日期",
        yaxis_title="累積損益 (TWD)",
        height=400
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # 交易記錄表
    st.subheader("📋 最近交易記錄")
    st.dataframe(
        trades_df.head(20),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("""
    ### 📝 尚無交易記錄
    
    交易記錄將在以下情況下產生:
    1. 執行回測後的模擬交易
    2. 實際交易記錄 (需手動輸入)
    
    您可以:
    - 前往「📈 回測分析」頁面執行回測
    - 在「⚙️ 系統設定」頁面設定交易參數
    """)

# 預測準確度
st.markdown("---")
st.header("🎯 預測準確度")

if not predictions_df.empty:
    # 計算準確度 (需要實際結果)
    st.info("預測準確度分析功能開發中...")
    
    # 顯示最近預測
    st.subheader("📊 最近預測記錄")
    st.dataframe(
        predictions_df.head(20),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("""
    ### 🔮 預測記錄
    
    預測記錄將在以下情況下產生:
    1. 使用「📊 即時預測」頁面進行預測
    2. 執行回測時的歷史預測
    
    預測記錄包含:
    - 預測日期
    - 預測方向 (看漲/看跌/盤整)
    - 信心度
    - 建議履約價
    - 實際結果 (待驗證)
    """)

# 統計摘要
st.markdown("---")
st.header("📊 統計摘要")

col1, col2 = st.columns(2)

with col1:
    st.subheader("市場統計")
    if not futures_df.empty:
        recent_df = futures_df.tail(30)
        stats = {
            '指標': [
                '30日平均價',
                '30日最高價',
                '30日最低價',
                '30日平均成交量',
                '30日波動率'
            ],
            '數值': [
                f"{recent_df['close'].mean():,.0f}",
                f"{recent_df['high'].max():,.0f}",
                f"{recent_df['low'].min():,.0f}",
                f"{recent_df['volume'].mean():,.0f}",
                f"{recent_df['close'].pct_change().std() * 100:.2f}%"
            ]
        }
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

with col2:
    st.subheader("系統狀態")
    system_stats = {
        '項目': [
            '歷史資料筆數',
            '交易記錄筆數',
            '預測記錄筆數',
            '最後更新時間',
            '資料完整度'
        ],
        '狀態': [
            f"{len(futures_df)} 筆",
            f"{len(trades_df)} 筆",
            f"{len(predictions_df)} 筆",
            futures_df.iloc[-1]['date'].strftime('%Y-%m-%d') if not futures_df.empty else 'N/A',
            "✅ 正常" if len(futures_df) > 30 else "⚠️ 資料不足"
        ]
    }
    st.dataframe(pd.DataFrame(system_stats), use_container_width=True, hide_index=True)
