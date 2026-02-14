"""
Discord 通知模組
"""
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from config.api_config import DISCORD_WEBHOOK_URL
from config.settings import ENABLE_DISCORD_NOTIFY, NOTIFY_ON_SIGNAL, NOTIFY_ON_ERROR
from src.utils.logger import get_logger
from src.utils.helpers import format_currency, format_percentage

logger = get_logger(__name__)


class DiscordNotifier:
    """Discord 通知器"""
    
    def __init__(self, webhook_url: str = None):
        """
        初始化通知器
        
        Args:
            webhook_url: Discord Webhook URL
        """
        self.webhook_url = webhook_url or DISCORD_WEBHOOK_URL
        self.enabled = ENABLE_DISCORD_NOTIFY and bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Discord 通知未啟用")
    
    def send_message(self, content: str, embeds: list = None) -> bool:
        """
        發送訊息到 Discord
        
        Args:
            content: 訊息內容
            embeds: 嵌入式訊息列表
        
        Returns:
            是否成功發送
        """
        if not self.enabled:
            logger.debug("Discord 通知已停用,跳過發送")
            return False
        
        try:
            payload = {"content": content}
            
            if embeds:
                payload["embeds"] = embeds
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("[OK] Discord 訊息已發送")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Discord 訊息發送失敗: {e}")
            return False
    
    def send_signal(
        self,
        action: str,
        market_data: Dict[str, Any],
        prediction: Dict[str, Any],
        advice: Dict[str, Any],
        options_analysis: Dict[str, Any]
    ):
        """
        發送交易訊號通知
        
        Args:
            action: 交易動作 (BUY_CALL/BUY_PUT/HOLD)
            market_data: 市場數據
            prediction: 預測結果
            advice: LLM 建議
            options_analysis: 選擇權分析
        """
        if not NOTIFY_ON_SIGNAL:
            return
        
        # 決定 emoji 和顏色
        if action == 'BUY_CALL':
            emoji = '🚀'
            color = 0x00FF00  # 綠色
            action_text = 'Buy Call (看漲)'
        elif action == 'BUY_PUT':
            emoji = '📉'
            color = 0xFF0000  # 紅色
            action_text = 'Buy Put (看跌)'
        else:
            emoji = '⏸️'
            color = 0xFFFF00  # 黃色
            action_text = '觀望'
        
        # 建立嵌入式訊息
        embed = {
            "title": f"{emoji} 選擇權進場訊號",
            "description": f"**策略: {action_text}**",
            "color": color,
            "fields": [
                {
                    "name": "📊 台指期現況",
                    "value": f"收盤: {market_data.get('close', 'N/A')}\n"
                            f"漲跌: {market_data.get('change', 'N/A')} ({market_data.get('change_pct', 'N/A')}%)\n"
                            f"成交量: {market_data.get('volume', 'N/A'):,}",
                    "inline": True
                },
                {
                    "name": "📈 AI 預測",
                    "value": f"方向: {prediction.get('direction', 'N/A')}\n"
                            f"信心度: {prediction.get('confidence', 0):.1%}\n"
                            f"預測漲跌: {prediction.get('predicted_change', 'N/A')}%",
                    "inline": True
                },
                {
                    "name": "💡 建議履約價",
                    "value": str(advice.get('strike_price', 'N/A')),
                    "inline": True
                },
                {
                    "name": "📉 波動率分析",
                    "value": f"IV: {options_analysis.get('avg_iv', 0):.2%}\n"
                            f"HV: {market_data.get('hv', 0):.2%}\n"
                            f"IV/HV: {options_analysis.get('iv_hv_ratio', 'N/A'):.2f}",
                    "inline": True
                },
                {
                    "name": "⚖️ 市場情緒",
                    "value": f"Put/Call Ratio: {options_analysis.get('pcr_volume', 'N/A'):.2f}\n"
                            f"情緒: {options_analysis.get('sentiment', 'N/A')}\n"
                            f"波動環境: {options_analysis.get('volatility_environment', 'N/A')}",
                    "inline": True
                },
                {
                    "name": "⚠️ 風險評估",
                    "value": advice.get('risk_level', 'medium').upper(),
                    "inline": True
                },
                {
                    "name": "📝 進場理由",
                    "value": advice.get('reasoning', '無'),
                    "inline": False
                },
                {
                    "name": "🛑 停損停利",
                    "value": advice.get('stop_loss', '無'),
                    "inline": False
                }
            ],
            "footer": {
                "text": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        # 加入警告訊息
        if advice.get('warnings'):
            embed["fields"].append({
                "name": "⚠️ 注意事項",
                "value": advice['warnings'],
                "inline": False
            })
        
        self.send_message("", embeds=[embed])
    
    def send_daily_report(
        self,
        market_summary: Dict[str, Any],
        performance: Dict[str, Any]
    ):
        """
        發送每日報告
        
        Args:
            market_summary: 市場摘要
            performance: 績效統計
        """
        embed = {
            "title": "📊 每日市場報告",
            "color": 0x0099FF,
            "fields": [
                {
                    "name": "市場摘要",
                    "value": f"台指期收盤: {market_summary.get('close', 'N/A')}\n"
                            f"漲跌: {market_summary.get('change', 'N/A')} ({market_summary.get('change_pct', 'N/A')}%)",
                    "inline": False
                },
                {
                    "name": "績效統計",
                    "value": f"總交易: {performance.get('total_trades', 0)} 筆\n"
                            f"勝率: {performance.get('win_rate', 0):.1%}\n"
                            f"累積損益: {format_currency(performance.get('total_pnl', 0))}",
                    "inline": False
                }
            ],
            "footer": {
                "text": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        self.send_message("", embeds=[embed])
    
    def send_error(self, error_message: str, details: str = None):
        """
        發送錯誤通知
        
        Args:
            error_message: 錯誤訊息
            details: 詳細資訊
        """
        if not NOTIFY_ON_ERROR:
            return
        
        embed = {
            "title": "❌ 系統錯誤",
            "description": error_message,
            "color": 0xFF0000,
            "footer": {
                "text": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        if details:
            embed["fields"] = [{
                "name": "詳細資訊",
                "value": details[:1000],  # 限制長度
                "inline": False
            }]
        
        self.send_message("", embeds=[embed])
    
    def test_connection(self) -> bool:
        """測試 Discord Webhook 連線"""
        logger.info("測試 Discord Webhook...")
        
        if not self.enabled:
            logger.warning("Discord 通知未啟用")
            return False
        
        embed = {
            "title": "✅ 測試訊息",
            "description": "台指期選擇權預測系統已啟動",
            "color": 0x00FF00,
            "footer": {
                "text": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        return self.send_message("", embeds=[embed])


# 測試程式碼
if __name__ == "__main__":
    notifier = DiscordNotifier()
    
    if notifier.test_connection():
        # 測試訊號通知
        test_market = {
            'close': 18500,
            'change': 150,
            'change_pct': 0.82,
            'volume': 120000,
            'hv': 0.18
        }
        
        test_prediction = {
            'direction': 'bullish',
            'confidence': 0.78,
            'predicted_change': 1.2
        }
        
        test_advice = {
            'action': 'BUY_CALL',
            'strike_price': 18600,
            'risk_level': 'medium',
            'reasoning': '技術指標轉強,波動率偏低',
            'stop_loss': '權利金跌破 50%'
        }
        
        test_options = {
            'pcr_volume': 0.85,
            'avg_iv': 0.20,
            'iv_hv_ratio': 0.90,
            'volatility_environment': 'low',
            'sentiment': 'bullish'
        }
        
        notifier.send_signal('BUY_CALL', test_market, test_prediction, test_advice, test_options)
