"""
方向性預測模型
使用 XGBoost 預測台指期未來走勢(漲/跌/盤整)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

from src.utils.logger import get_logger
from config.model_config import DIRECTION_MODEL_PARAMS, FEATURE_COLUMNS
from config.settings import MODEL_DIR

logger = get_logger(__name__)


class DirectionPredictor:
    """方向性預測模型"""
    
    def __init__(self, model_path: str = None):
        """
        初始化預測器
        
        Args:
            model_path: 模型檔案路徑
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or f"{MODEL_DIR}/direction_model.pkl"
        self.scaler_path = f"{MODEL_DIR}/direction_scaler.pkl"
        
        # 確保模型目錄存在
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info("方向性預測器初始化完成")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        準備特徵資料
        
        Args:
            df: 包含技術指標的 DataFrame
        
        Returns:
            特徵 DataFrame
        """
        # 選擇特徵欄位
        feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        if not feature_cols:
            logger.warning("未找到任何特徵欄位,使用預設欄位")
            feature_cols = ['close', 'volume', 'rsi', 'macd', 'atr']
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        features = df[feature_cols].copy()
        
        # 處理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"準備特徵完成: {len(feature_cols)} 個特徵")
        return features
    
    def create_labels(self, df: pd.DataFrame, forward_days: int = 1, threshold: float = 0.5) -> pd.Series:
        """
        建立標籤(漲/跌/盤整)
        
        Args:
            df: 資料 DataFrame
            forward_days: 預測未來幾天
            threshold: 漲跌判斷閾值(%)
        
        Returns:
            標籤 Series (0: 跌, 1: 盤整, 2: 漲)
        """
        # 計算未來報酬率
        future_return = (df['close'].shift(-forward_days) - df['close']) / df['close'] * 100
        
        # 建立標籤
        labels = pd.Series(1, index=df.index)  # 預設為盤整
        labels[future_return > threshold] = 2   # 漲
        labels[future_return < -threshold] = 0  # 跌
        
        # 移除最後幾筆(無法計算未來報酬)
        labels.iloc[-forward_days:] = np.nan
        
        logger.info(f"標籤分布 - 跌: {(labels == 0).sum()}, 盤整: {(labels == 1).sum()}, 漲: {(labels == 2).sum()}")
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        forward_days: int = 1,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        訓練模型
        
        Args:
            df: 訓練資料
            test_size: 測試集比例
            forward_days: 預測未來幾天
            threshold: 漲跌判斷閾值
        
        Returns:
            訓練結果字典
        """
        logger.info("開始訓練方向性預測模型...")
        
        # 準備特徵與標籤
        features = self.prepare_features(df)
        labels = self.create_labels(df, forward_days, threshold)
        
        # 移除缺失值
        valid_idx = ~labels.isna()
        X = features[valid_idx]
        y = labels[valid_idx]
        
        # 時間序列分割(避免資料洩漏)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練 XGBoost
        self.model = xgb.XGBClassifier(**DIRECTION_MODEL_PARAMS)
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # 預測
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # 評估
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"訓練集準確率: {train_acc:.2%}")
        logger.info(f"測試集準確率: {test_acc:.2%}")
        
        # 儲存模型
        self.save_model()
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        預測方向
        
        Args:
            df: 包含最新資料的 DataFrame
        
        Returns:
            (預測方向, 信心度)
            方向: 0=跌, 1=盤整, 2=漲
        """
        if self.model is None:
            self.load_model()
        
        # 準備特徵
        features = self.prepare_features(df)
        
        # 使用最新一筆資料
        X = features.iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        # 預測
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = probabilities[prediction]
        
        direction_map = {0: '跌', 1: '盤整', 2: '漲'}
        logger.info(f"預測方向: {direction_map[prediction]}, 信心度: {confidence:.2%}")
        
        return int(prediction), float(confidence)
    
    def save_model(self):
        """儲存模型"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"模型已儲存至: {self.model_path}")
        except Exception as e:
            logger.error(f"儲存模型失敗: {e}")
    
    def load_model(self):
        """載入模型"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"模型已載入: {self.model_path}")
        except FileNotFoundError:
            logger.warning("模型檔案不存在,請先訓練模型")
        except Exception as e:
            logger.error(f"載入模型失敗: {e}")


# 測試程式碼
if __name__ == "__main__":
    from src.data.database import Database
    from src.features.technical import add_all_technical_indicators
    
    # 載入資料
    with Database() as db:
        df = db.get_futures_data()
    
    if not df.empty:
        # 計算技術指標
        df_with_indicators = add_all_technical_indicators(df)
        
        # 訓練模型
        predictor = DirectionPredictor()
        results = predictor.train(df_with_indicators)
        
        print("\n=== 訓練結果 ===")
        print(f"訓練集準確率: {results['train_accuracy']:.2%}")
        print(f"測試集準確率: {results['test_accuracy']:.2%}")
        
        print("\n=== 特徵重要性 (Top 10) ===")
        importance = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in importance[:10]:
            print(f"{feat}: {imp:.4f}")
        
        # 測試預測
        direction, confidence = predictor.predict(df_with_indicators)
        direction_map = {0: '跌', 1: '盤整', 2: '漲'}
        print(f"\n=== 最新預測 ===")
        print(f"方向: {direction_map[direction]}")
        print(f"信心度: {confidence:.2%}")
