"""
集成預測系統
整合方向性預測、波動率預測與 LLM 建議
"""
from typing import Dict, Any, Tuple
import pandas as pd
from datetime import datetime

from src.models.direction_model import DirectionPredictor
from src.models.volatility_model import VolatilityPredictor
from src.models.llm_advisor import LLMAdvisor
from src.features.technical import add_all_technical_indicators
from src.features.options_metrics import calculate_historical_volatility
from src.utils.logger import get_logger
from config.model_config import PREDICTION_THRESHOLDS

logger = get_logger(__name__)


class EnsemblePredictor:
    """集成預測系統"""
    
    def __init__(self):
        """初始化集成預測器"""
        self.direction_predictor = DirectionPredictor()
        self.volatility_predictor = VolatilityPredictor()
        self.llm_advisor = LLMAdvisor()
        
        logger.info("集成預測系統初始化完成")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        綜合預測
        
        Args:
            df: 包含歷史資料的 DataFrame
        
        Returns:
            預測結果字典
        """
        logger.info("開始執行集成預測...")
        
        # 確保有技術指標
        if 'rsi' not in df.columns:
            df = add_all_technical_indicators(df)
        
        # 1. 方向性預測
        try:
            direction, direction_confidence = self.direction_predictor.predict(df)
            direction_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            direction_str = direction_map[direction]
        except Exception as e:
            logger.error(f"方向性預測失敗: {e}")
            direction_str = 'neutral'
            direction_confidence = 0.5
        
        # 2. 波動率預測
        try:
            predicted_vol, current_vol = self.volatility_predictor.predict(df)
            if predicted_vol is None:
                predicted_vol = current_vol = 0.2  # 預設值
        except Exception as e:
            logger.error(f"波動率預測失敗: {e}")
            predicted_vol = current_vol = 0.2
        
        # 3. 計算選擇權指標
        latest = df.iloc[-1]
        hv = calculate_historical_volatility(df)
        
        # 模擬選擇權分析(簡化版)
        options_analysis = {
            'pcr_volume': 0.9,  # 假設值
            'avg_iv': current_vol * 1.1,  # 假設 IV 略高於 HV
            'iv_hv_ratio': 1.1,
            'volatility_environment': self._classify_volatility(current_vol),
            'sentiment': 'neutral',
            'max_pain': latest['close']
        }
        
        # 4. 整合預測結果
        prediction = {
            'direction': direction_str,
            'confidence': direction_confidence,
            'predicted_change': self._estimate_change(direction_str, direction_confidence),
            'volatility': {
                'current': current_vol,
                'predicted': predicted_vol,
                'trend': 'rising' if predicted_vol > current_vol * 1.05 else 'falling' if predicted_vol < current_vol * 0.95 else 'stable'
            }
        }
        
        # 5. 市場資料
        market_data = {
            'close': latest['close'],
            'change': latest['close'] - latest['open'],
            'change_pct': (latest['close'] - latest['open']) / latest['open'] * 100,
            'volume': latest['volume'],
            'rsi': latest.get('rsi', 50),
            'macd': latest.get('macd', 0),
            'bb_position': self._get_bb_position(latest),
            'atr': latest.get('atr', 100),
            'hv': hv.iloc[-1] if not hv.empty else current_vol
        }
        
        # 6. LLM 策略建議
        try:
            llm_advice = self.llm_advisor.get_trading_advice(
                market_data,
                prediction,
                options_analysis
            )
        except Exception as e:
            logger.error(f"LLM 建議失敗: {e}")
            llm_advice = {
                'action': 'HOLD',
                'reasoning': 'LLM 服務暫時不可用',
                'risk_level': 'medium',
                'strike_price': None,
                'stop_loss': None,
                'warnings': 'LLM 連線失敗,請檢查 Ollama 服務'
            }
        
        # 7. 綜合結果
        result = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'prediction': prediction,
            'options_analysis': options_analysis,
            'llm_advice': llm_advice,
            'final_recommendation': self._make_final_recommendation(
                prediction,
                options_analysis,
                llm_advice
            )
        }
        
        logger.info(f"集成預測完成: {result['final_recommendation']['action']}")
        
        return result
    
    def _estimate_change(self, direction: str, confidence: float) -> float:
        """估算預期漲跌幅"""
        if direction == 'bullish':
            return 1.0 * confidence
        elif direction == 'bearish':
            return -1.0 * confidence
        else:
            return 0.0
    
    def _classify_volatility(self, volatility: float) -> str:
        """分類波動率環境"""
        if volatility > 0.25:
            return 'high'
        elif volatility > 0.15:
            return 'normal'
        else:
            return 'low'
    
    def _get_bb_position(self, latest: pd.Series) -> str:
        """取得布林通道位置"""
        if 'bb_upper' in latest and 'bb_lower' in latest:
            if latest['close'] > latest['bb_upper']:
                return 'above_upper'
            elif latest['close'] < latest['bb_lower']:
                return 'below_lower'
            else:
                return 'middle'
        return 'middle'
    
    def _make_final_recommendation(
        self,
        prediction: Dict,
        options_analysis: Dict,
        llm_advice: Dict
    ) -> Dict[str, Any]:
        """
        綜合所有資訊做出最終建議
        
        Args:
            prediction: 預測結果
            options_analysis: 選擇權分析
            llm_advice: LLM 建議
        
        Returns:
            最終建議字典
        """
        # 使用 LLM 建議作為主要依據
        action = llm_advice.get('action', 'HOLD')
        
        # 檢查信心度閾值
        if prediction['confidence'] < PREDICTION_THRESHOLDS['min_confidence']:
            action = 'HOLD'
            reason = f"信心度過低 ({prediction['confidence']:.1%})"
        else:
            reason = llm_advice.get('reasoning', '')
        
        return {
            'action': action,
            'reason': reason,
            'confidence': prediction['confidence'],
            'risk_level': llm_advice.get('risk_level', 'medium'),
            'strike_price': llm_advice.get('strike_price'),
            'stop_loss': llm_advice.get('stop_loss'),
            'warnings': llm_advice.get('warnings', '')
        }


# 測試程式碼
if __name__ == "__main__":
    from src.data.database import Database
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if not df.empty:
        # 執行集成預測
        ensemble = EnsemblePredictor()
        result = ensemble.predict(df)
        
        print("\n=== 集成預測結果 ===")
        print(f"時間: {result['timestamp']}")
        print(f"\n市場資料:")
        print(f"  收盤價: {result['market_data']['close']:.0f}")
        print(f"  漲跌: {result['market_data']['change']:+.0f} ({result['market_data']['change_pct']:+.2f}%)")
        print(f"  RSI: {result['market_data']['rsi']:.1f}")
        
        print(f"\n方向預測:")
        print(f"  方向: {result['prediction']['direction']}")
        print(f"  信心度: {result['prediction']['confidence']:.1%}")
        print(f"  預期漲跌: {result['prediction']['predicted_change']:+.2f}%")
        
        print(f"\n波動率:")
        print(f"  當前: {result['prediction']['volatility']['current']:.2%}")
        print(f"  預測: {result['prediction']['volatility']['predicted']:.2%}")
        print(f"  趨勢: {result['prediction']['volatility']['trend']}")
        
        print(f"\nLLM 建議:")
        print(f"  動作: {result['llm_advice']['action']}")
        print(f"  理由: {result['llm_advice']['reasoning']}")
        print(f"  風險: {result['llm_advice']['risk_level']}")
        
        print(f"\n最終建議:")
        print(f"  動作: {result['final_recommendation']['action']}")
        print(f"  理由: {result['final_recommendation']['reason']}")
