"""
回測分析頁面
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
from src.models.ensemble import EnsemblePredictor
from src.backtest.engine import BacktestEngine

st.set_page_config(page_title="回測分析", page_icon="📈", layout="wide")

st.title("📈 回測分析")
st.markdown("---")

# 側邊欄設定
st.sidebar.header("回測參數")

# 回測期間選擇
backtest_days = st.sidebar.slider("回測天數", 30, 180, 90)
holding_period = st.sidebar.slider("持有天數", 1, 10, 5)
prediction_interval = st.sidebar.slider("預測間隔(天)", 1, 10, 5)

# 執行回測按鈕
if st.sidebar.button("🚀 執行回測", type="primary"):
    with st.spinner("正在執行回測..."):
        try:
            # 載入資料
            with Database() as db:
                df = db.get_futures_data()
            
            if df.empty:
                st.error("無歷史資料,請先執行 daily_update.py")
                st.stop()
            
            # 限制回測天數
            df = df.tail(backtest_days + 100)  # 多取一些資料用於預測
            
            # 生成預測
            ensemble = EnsemblePredictor()
            predictions = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_predictions = (len(df) - 100) // prediction_interval
            
            for i, idx in enumerate(range(100, len(df), prediction_interval)):
                subset = df.iloc[:idx]
                try:
                    result = ensemble.predict(subset)
                    predictions.append({
                        'date': subset.iloc[-1]['date'],
                        'direction': result['prediction']['direction'],
                        'confidence': result['prediction']['confidence']
                    })
                    
                    progress = (i + 1) / total_predictions
                    progress_bar.progress(progress)
                    status_text.text(f"生成預測中... {i+1}/{total_predictions}")
                except:
                    continue
            
            predictions_df = pd.DataFrame(predictions)
            
            # 執行回測
            status_text.text("執行回測中...")
            engine = BacktestEngine()
            results = engine.run_backtest(predictions_df, df, holding_period=holding_period)
            
            # 儲存結果到 session state
            st.session_state['backtest_results'] = results
            st.session_state['backtest_df'] = df
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ 回測完成! 共 {len(results['trades'])} 筆交易")
            
        except Exception as e:
            st.error(f"回測失敗: {e}")
            import traceback
            st.code(traceback.format_exc())

# 顯示回測結果
if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    metrics = results['metrics']
    
    # 績效總覽
    st.header("📊 績效總覽")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "總報酬率",
            f"{metrics['total_return']:.2%}",
            delta=f"${results['final_capital'] - results['initial_capital']:,.0f}"
        )
    
    with col2:
        st.metric(
            "勝率",
            f"{metrics['win_rate']:.2%}",
            delta=f"{metrics['winning_trades']}/{metrics['total_trades']}"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="年化"
        )
    
    with col4:
        st.metric(
            "最大回撤",
            f"{metrics['max_drawdown']:.2%}",
            delta="風險指標",
            delta_color="inverse"
        )
    
    # 詳細指標
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("交易統計")
        stats_df = pd.DataFrame({
            '指標': [
                '總交易次數',
                '獲利次數',
                '虧損次數',
                '平均報酬率',
                '獲利因子'
            ],
            '數值': [
                f"{metrics['total_trades']}",
                f"{metrics['winning_trades']}",
                f"{metrics['losing_trades']}",
                f"{metrics['avg_return']:.2%}",
                f"{metrics['profit_factor']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("損益分析")
        pnl_df = pd.DataFrame({
            '類型': ['平均獲利', '平均虧損', '獲利/虧損比'],
            '金額': [
                f"${metrics['avg_profit']:,.0f}",
                f"${metrics['avg_loss']:,.0f}",
                f"{metrics['avg_profit']/metrics['avg_loss']:.2f}" if metrics['avg_loss'] > 0 else "N/A"
            ]
        })
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)
    
    # 權益曲線
    st.markdown("---")
    st.subheader("💰 權益曲線")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        y=results['equity_curve'],
        mode='lines',
        name='權益',
        line=dict(color='#00D9FF', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    fig_equity.add_hline(
        y=results['initial_capital'],
        line_dash="dash",
        line_color="gray",
        annotation_text="初始資金"
    )
    
    fig_equity.update_layout(
        title="權益曲線變化",
        xaxis_title="交易次數",
        yaxis_title="權益 (TWD)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # 交易記錄
    st.markdown("---")
    st.subheader("📋 交易記錄")
    
    if results['trades']:
        trades_data = []
        for trade in results['trades']:
            trades_data.append({
                '進場日期': trade.entry_date,
                '出場日期': trade.exit_date,
                '方向': trade.direction.upper(),
                '履約價': f"{trade.strike_price:,.0f}",
                '進場權利金': f"{trade.entry_price:.2f}",
                '出場權利金': f"{trade.exit_price:.2f}",
                '損益': f"${trade.profit_loss:+,.0f}",
                '報酬率': f"{trade.return_pct:+.1%}",
                '持有天數': trade.holding_days
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # 顯示最近 20 筆交易
        st.dataframe(
            trades_df.tail(20),
            use_container_width=True,
            hide_index=True
        )
        
        # 下載完整交易記錄
        csv = trades_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載完整交易記錄 (CSV)",
            data=csv,
            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # 交易分布圖
        st.markdown("---")
        st.subheader("📊 交易分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 方向分布
            direction_counts = trades_df['方向'].value_counts()
            fig_direction = px.pie(
                values=direction_counts.values,
                names=direction_counts.index,
                title="交易方向分布",
                color_discrete_sequence=['#00D9FF', '#FF6B9D']
            )
            st.plotly_chart(fig_direction, use_container_width=True)
        
        with col2:
            # 損益分布
            pnl_values = [trade.profit_loss for trade in results['trades']]
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(
                x=pnl_values,
                nbinsx=20,
                marker_color='#00D9FF',
                name='損益分布'
            ))
            fig_pnl.update_layout(
                title="損益分布圖",
                xaxis_title="損益 (TWD)",
                yaxis_title="次數"
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    
    else:
        st.info("無交易記錄")

else:
    # 初始說明
    st.info("""
    ### 🎯 如何使用回測功能
    
    1. **設定回測參數**
       - 回測天數: 選擇要回測的歷史資料範圍
       - 持有天數: 每筆交易的持有期間
       - 預測間隔: 每隔幾天進行一次預測
    
    2. **執行回測**
       - 點擊左側「🚀 執行回測」按鈕
       - 系統將自動生成歷史預測並模擬交易
    
    3. **查看結果**
       - 績效總覽: 總報酬率、勝率、Sharpe Ratio
       - 權益曲線: 資金變化趨勢
       - 交易記錄: 詳細的進出場記錄
    
    ⚠️ **注意事項**
    - 回測結果僅供參考,不代表未來績效
    - 實際交易需考慮滑價、手續費等成本
    - 建議多次回測以驗證策略穩定性
    """)
    
    # 顯示範例圖表
    st.markdown("---")
    st.subheader("📈 範例: 權益曲線")
    
    # 生成範例資料
    example_equity = [100000]
    for i in range(50):
        change = example_equity[-1] * (0.02 if i % 3 != 0 else -0.01)
        example_equity.append(example_equity[-1] + change)
    
    fig_example = go.Figure()
    fig_example.add_trace(go.Scatter(
        y=example_equity,
        mode='lines',
        line=dict(color='#00D9FF', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    fig_example.update_layout(
        title="範例權益曲線 (示意圖)",
        xaxis_title="交易次數",
        yaxis_title="權益 (TWD)",
        height=300
    )
    
    st.plotly_chart(fig_example, use_container_width=True)
