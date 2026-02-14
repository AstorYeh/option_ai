"""
LLM 策略顧問模組
使用 Local LLM (Ollama) 提供交易建議
"""
import requests
import json
from typing import Dict, Any, Optional
from config.api_config import OLLAMA_API_URL, OLLAMA_MODEL
from config.model_config import LLM_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMAdvisor:
    """LLM 策略顧問"""
    
    def __init__(self, api_url: str = None, model: str = None):
        """
        初始化 LLM 顧問
        
        Args:
            api_url: Ollama API URL
            model: 模型名稱
        """
        self.api_url = api_url or OLLAMA_API_URL
        self.model = model or OLLAMA_MODEL
        self.system_prompt = LLM_CONFIG['system_prompt']
        
        logger.info(f"LLM 顧問初始化: {self.model}")
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        呼叫 Ollama API
        
        Args:
            prompt: 提示詞
        
        Returns:
            LLM 回應
        """
        try:
            url = f"{self.api_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_CONFIG['temperature'],
                    "num_predict": LLM_CONFIG['max_tokens']
                }
            }
            
            logger.debug(f"呼叫 Ollama API: {self.model}")
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 呼叫失敗: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM 處理錯誤: {e}")
            return None
    
    def get_trading_advice(
        self,
        market_data: Dict[str, Any],
        prediction: Dict[str, Any],
        options_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        取得交易建議
        
        Args:
            market_data: 市場數據
            prediction: 模型預測結果
            options_analysis: 選擇權分析結果
        
        Returns:
            交易建議字典
        """
        logger.info("請求 LLM 交易建議...")
        
        # 建立提示詞
        prompt = self._build_prompt(market_data, prediction, options_analysis)
        
        # 呼叫 LLM
        response = self._call_ollama(prompt)
        
        if not response:
            logger.error("LLM 未回應,使用預設建議")
            return self._get_default_advice(prediction)
        
        # 解析回應
        advice = self._parse_response(response, prediction)
        
        logger.info(f"[OK] LLM 建議: {advice.get('action', 'UNKNOWN')}")
        
        return advice
    
    def _build_prompt(
        self,
        market_data: Dict[str, Any],
        prediction: Dict[str, Any],
        options_analysis: Dict[str, Any]
    ) -> str:
        """建立提示詞"""
        
        prompt = f"""
當前市場狀況分析:

【台指期現況】
- 收盤價: {market_data.get('close', 'N/A')}
- 日漲跌: {market_data.get('change', 'N/A')} ({market_data.get('change_pct', 'N/A')}%)
- 成交量: {market_data.get('volume', 'N/A')}

【技術指標】
- RSI: {market_data.get('rsi', 'N/A'):.1f}
- MACD: {market_data.get('macd', 'N/A'):.1f}
- 布林通道位置: {market_data.get('bb_position', 'N/A')}
- ATR: {market_data.get('atr', 'N/A'):.1f}

【AI 模型預測】
- 方向預測: {prediction.get('direction', 'N/A')}
- 信心度: {prediction.get('confidence', 0):.1%}
- 預測漲跌幅: {prediction.get('predicted_change', 'N/A')}%

【選擇權市場分析】
- Put/Call Ratio: {options_analysis.get('pcr_volume', 'N/A'):.2f}
- 隱含波動率: {options_analysis.get('avg_iv', 'N/A'):.2%}
- 歷史波動率: {market_data.get('hv', 'N/A'):.2%}
- IV/HV 比值: {options_analysis.get('iv_hv_ratio', 'N/A'):.2f}
- 波動率環境: {options_analysis.get('volatility_environment', 'N/A')}
- 市場情緒: {options_analysis.get('sentiment', 'N/A')}
- 最大痛點: {options_analysis.get('max_pain', 'N/A')}

請根據以上資訊,提供專業的選擇權買方策略建議。

請按照以下格式回答:
1. 策略建議: [BUY_CALL/BUY_PUT/HOLD]
2. 建議履約價: [具體價位或 N/A]
3. 風險評估: [低/中/高]
4. 進場理由: [簡要說明,50字以內]
5. 停損停利: [具體建議]
6. 注意事項: [風險提醒]
"""
        
        return prompt
    
    def _parse_response(self, response: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 LLM 回應
        
        Args:
            response: LLM 回應文字
            prediction: 模型預測(作為備援)
        
        Returns:
            結構化建議
        """
        advice = {
            'action': 'HOLD',
            'strike_price': None,
            'risk_level': 'medium',
            'reasoning': '',
            'stop_loss': '',
            'take_profit': '',
            'warnings': '',
            'raw_response': response
        }
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 策略建議
                if '策略建議' in line or '1.' in line:
                    if 'BUY_CALL' in line.upper() or '買進CALL' in line or '看漲' in line:
                        advice['action'] = 'BUY_CALL'
                    elif 'BUY_PUT' in line.upper() or '買進PUT' in line or '看跌' in line:
                        advice['action'] = 'BUY_PUT'
                    elif 'HOLD' in line.upper() or '觀望' in line or '等待' in line:
                        advice['action'] = 'HOLD'
                
                # 履約價
                elif '建議履約價' in line or '2.' in line:
                    # 嘗試提取數字
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        advice['strike_price'] = int(numbers[0])
                
                # 風險評估
                elif '風險評估' in line or '3.' in line:
                    if '低' in line:
                        advice['risk_level'] = 'low'
                    elif '高' in line:
                        advice['risk_level'] = 'high'
                    else:
                        advice['risk_level'] = 'medium'
                
                # 進場理由
                elif '進場理由' in line or '4.' in line:
                    advice['reasoning'] = line.split(':', 1)[-1].strip()
                
                # 停損停利
                elif '停損停利' in line or '5.' in line:
                    advice['stop_loss'] = line.split(':', 1)[-1].strip()
                
                # 注意事項
                elif '注意事項' in line or '6.' in line:
                    advice['warnings'] = line.split(':', 1)[-1].strip()
            
            # 若未成功解析,使用模型預測作為備援
            if advice['action'] == 'HOLD' and prediction.get('confidence', 0) > 0.7:
                direction = prediction.get('direction', 'neutral')
                if direction == 'bullish':
                    advice['action'] = 'BUY_CALL'
                elif direction == 'bearish':
                    advice['action'] = 'BUY_PUT'
            
        except Exception as e:
            logger.error(f"解析 LLM 回應失敗: {e}")
        
        return advice
    
    def _get_default_advice(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """取得預設建議(當 LLM 失敗時)"""
        
        confidence = prediction.get('confidence', 0)
        direction = prediction.get('direction', 'neutral')
        
        if confidence < 0.65:
            action = 'HOLD'
            reasoning = '信心度不足,建議觀望'
        elif direction == 'bullish':
            action = 'BUY_CALL'
            reasoning = '模型預測看漲'
        elif direction == 'bearish':
            action = 'BUY_PUT'
            reasoning = '模型預測看跌'
        else:
            action = 'HOLD'
            reasoning = '方向不明確,建議觀望'
        
        return {
            'action': action,
            'strike_price': None,
            'risk_level': 'medium',
            'reasoning': reasoning,
            'stop_loss': '權利金跌破 50%',
            'take_profit': '權利金翻倍',
            'warnings': 'LLM 服務不可用,使用預設建議',
            'raw_response': None
        }
    
    def test_connection(self) -> bool:
        """測試 Ollama 連線"""
        logger.info("測試 Ollama 連線...")
        
        try:
            url = f"{self.api_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            logger.info(f"可用模型: {model_names}")
            
            if self.model in model_names or any(self.model in name for name in model_names):
                logger.info(f"[OK] 模型 {self.model} 可用")
                return True
            else:
                logger.warning(f"[WARN] 模型 {self.model} 不在可用清單中")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Ollama 連線失敗: {e}")
            return False


# 測試程式碼
if __name__ == "__main__":
    advisor = LLMAdvisor()
    
    # 測試連線
    if advisor.test_connection():
        # 測試建議生成
        test_market_data = {
            'close': 18500,
            'change': 150,
            'change_pct': 0.82,
            'volume': 120000,
            'rsi': 65,
            'macd': 50,
            'bb_position': 'upper',
            'atr': 200,
            'hv': 0.18
        }
        
        test_prediction = {
            'direction': 'bullish',
            'confidence': 0.78,
            'predicted_change': 1.2
        }
        
        test_options = {
            'pcr_volume': 0.85,
            'avg_iv': 0.20,
            'iv_hv_ratio': 0.90,
            'volatility_environment': 'low',
            'sentiment': 'bullish',
            'max_pain': 18400
        }
        
        advice = advisor.get_trading_advice(test_market_data, test_prediction, test_options)
        
        print("\n=== LLM 交易建議 ===")
        print(f"策略: {advice['action']}")
        print(f"履約價: {advice['strike_price']}")
        print(f"風險: {advice['risk_level']}")
        print(f"理由: {advice['reasoning']}")
        print(f"\n完整回應:\n{advice['raw_response']}")
