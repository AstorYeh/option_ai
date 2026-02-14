"""
每日市場分析報告生成器
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import Database
from src.models.ensemble import EnsemblePredictor
from src.features.technical import add_all_technical_indicators
from src.utils.logger import get_logger

# Discord 通知為可選功能
try:
    from src.notifications.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("Discord 通知模組未安裝,將跳過通知功能")

logger = get_logger(__name__)


def generate_market_report():
    """生成每日市場分析報告"""
    logger.info("=== 開始生成每日市場分析報告 ===")
    
    try:
        # 載入資料
        with Database() as db:
            df = db.get_futures_data()
        
        if df.empty or len(df) < 30:
            logger.error("資料不足,無法生成報告")
            return None
        
        # 計算技術指標
        df = add_all_technical_indicators(df)
        
        # 取得最新資料
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 計算變化
        price_change = latest['close'] - prev['close']
        price_change_pct = (price_change / prev['close']) * 100
        
        # 計算統計數據
        recent_30 = df.tail(30)
        avg_price_30 = recent_30['close'].mean()
        volatility_30 = recent_30['close'].pct_change().std() * 100
        
        # 執行 AI 預測
        ensemble = EnsemblePredictor()
        prediction = ensemble.predict(df)
        
        # 生成報告
        report = f"""
╔══════════════════════════════════════════════════════════╗
║          台指期選擇權 - 每日市場分析報告                ║
║          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                      ║
╚══════════════════════════════════════════════════════════╝

【市場概況】
[Market] 台指期收盤價: {latest['close']:,.0f} 點
[UP/DN] 漲跌: {price_change:+.0f} 點 ({price_change_pct:+.2f}%)
[RANGE] 最高/最低: {latest['high']:,.0f} / {latest['low']:,.0f}
[VOL] 成交量: {latest['volume']:,.0f}

【技術指標】
RSI(14): {latest['rsi']:.1f} {'(超買)' if latest['rsi'] > 70 else '(超賣)' if latest['rsi'] < 30 else '(中性)'}
MACD: {latest['macd']:.2f}
布林通道: 上軌 {latest['bb_upper']:,.0f} / 中軌 {latest['bb_middle']:,.0f} / 下軌 {latest['bb_lower']:,.0f}
ATR(14): {latest['atr']:.2f} (波動度)

【統計分析】
30日平均價: {avg_price_30:,.0f} 點
30日波動率: {volatility_30:.2f}%
當前位置: {'高於' if latest['close'] > avg_price_30 else '低於'}平均價 {abs(latest['close'] - avg_price_30):.0f} 點

【AI 預測建議】
[AI] 方向預測: {prediction['prediction']['direction'].upper()}
[CONF] 信心度: {prediction['prediction']['confidence']:.1%}
[TARGET] 預期變化: {prediction['prediction']['predicted_change']:+.2f}%

[LLM] LLM 建議:
動作: {prediction['llm_advice']['action']}
理由: {prediction['llm_advice']['reasoning'][:200]}...
風險等級: {prediction['llm_advice']['risk_level']}

【最終建議】
[OK] 建議動作: {prediction['final_recommendation']['action']}
[NOTE] 理由: {prediction['final_recommendation']['reason'][:200]}...
[TARGET] 信心度: {prediction['final_recommendation']['confidence']:.1%}
[WARN] 風險等級: {prediction['final_recommendation']['risk_level']}

【市場情緒】
{_get_market_sentiment(latest, recent_30)}

【注意事項】
[WARN] 本報告僅供參考,不構成投資建議
[WARN] 實際交易請自行評估風險
[WARN] 選擇權交易具有高風險,請謹慎操作

═══════════════════════════════════════════════════════════
報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
資料來源: FinMind API
AI 模型: XGBoost + LSTM + LLM (Qwen2.5:3B)
═══════════════════════════════════════════════════════════
"""
        
        logger.info("[OK] 市場分析報告生成完成")
        return report
        
    except Exception as e:
        logger.error(f"生成報告失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _get_market_sentiment(latest, recent_df):
    """分析市場情緒"""
    sentiment_lines = []
    
    # RSI 情緒
    if latest['rsi'] > 70:
        sentiment_lines.append("[WARN] RSI 顯示市場過熱,可能面臨回調壓力")
    elif latest['rsi'] < 30:
        sentiment_lines.append("[OK] RSI 顯示市場超賣,可能出現反彈機會")
    else:
        sentiment_lines.append("[INFO] RSI 處於中性區間,市場情緒平穩")
    
    # MACD 情緒
    if latest['macd'] > latest['macd_signal']:
        sentiment_lines.append("[UP] MACD 呈現多頭排列,短期趨勢向上")
    else:
        sentiment_lines.append("[DN] MACD 呈現空頭排列,短期趨勢向下")
    
    # 布林通道位置
    bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
    if bb_position > 0.8:
        sentiment_lines.append("[WARN] 價格接近布林通道上軌,注意回調風險")
    elif bb_position < 0.2:
        sentiment_lines.append("[OK] 價格接近布林通道下軌,可能有支撐")
    else:
        sentiment_lines.append("[--] 價格位於布林通道中間,區間震盪")
    
    # 成交量分析
    avg_volume = recent_df['volume'].mean()
    if latest['volume'] > avg_volume * 1.5:
        sentiment_lines.append("[VOL+] 成交量顯著放大,市場關注度高")
    elif latest['volume'] < avg_volume * 0.5:
        sentiment_lines.append("[VOL-] 成交量萎縮,市場觀望氣氛濃厚")
    
    return "\n".join(sentiment_lines)


def send_report_to_discord(report):
    """發送報告到 Discord"""
    if not DISCORD_AVAILABLE:
        logger.warning("Discord 通知功能未啟用")
        return False
    
    try:
        notifier = DiscordNotifier()
        
        # Discord 訊息格式化
        discord_message = f"""
**📊 台指期選擇權 - 每日市場分析報告**
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

```
{report}
```
"""
        
        notifier.send_message(discord_message)
        logger.info("[OK] 報告已發送至 Discord")
        return True
        
    except Exception as e:
        logger.warning(f"發送 Discord 通知失敗: {e}")
        return False


def save_report_to_file(report):
    """儲存報告到檔案"""
    try:
        # 建立報告目錄
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        # 儲存報告
        filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = report_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[OK] 報告已儲存至: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"儲存報告失敗: {e}")
        return None


if __name__ == "__main__":
    # 生成報告
    report = generate_market_report()
    
    if report:
        # 顯示報告
        print(report)
        
        # 儲存報告
        filepath = save_report_to_file(report)
        
        # 發送到 Discord (如果有設定)
        send_report_to_discord(report)
        
        print("\n" + "="*60)
        print("[OK] 每日市場分析報告生成完成!")
        if filepath:
            print(f"[FILE] 報告已儲存至: {filepath}")
        print("="*60)
    else:
        print("[ERROR] 報告生成失敗")
