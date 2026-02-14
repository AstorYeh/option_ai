"""
模型配置與超參數
"""

# 方向性預測模型配置
DIRECTION_MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "lstm": {
        "units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "lookback_days": 20
    }
}

# 波動率預測模型配置
VOLATILITY_MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "garch": {
        "p": 1,
        "q": 1,
        "window": 252  # 一年交易日
    }
}

# 特徵工程配置
FEATURE_CONFIG = {
    "technical_indicators": {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "atr_period": 14,
        "ma_periods": [5, 20, 60]
    },
    "volatility_metrics": {
        "hv_window": 20,  # 歷史波動率計算窗口
        "iv_percentile_window": 252  # IV 百分位計算窗口
    }
}

# 預測閾值
PREDICTION_THRESHOLDS = {
    "buy_call_confidence": 0.65,  # Buy Call 最低信心度
    "buy_put_confidence": 0.65,   # Buy Put 最低信心度
    "volatility_low_threshold": 0.85,  # IV/HV 比值低於此值視為低波動
    "price_change_threshold": 0.01,  # 價格變動閾值(1%)
    "min_confidence": 0.6  # 最低信心度
}

# LLM 配置
LLM_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 500,
    "system_prompt": """你是一位專業的選擇權交易顧問,專注於買方策略(Buy Call/Put)。
請根據提供的市場數據與模型預測,給出明確的交易建議。
回答格式:
1. 策略建議: [Buy Call/Buy Put/觀望]
2. 建議履約價: [具體價位]
3. 風險評估: [低/中/高]
4. 進場理由: [簡要說明]
5. 停損停利: [具體建議]
"""
}

# 訓練配置
TRAINING_CONFIG = {
    "train_test_split": 0.8,  # 訓練集比例
    "validation_split": 0.2,  # 驗證集比例
    "cross_validation_folds": 5,
    "early_stopping_patience": 10
}

# 回測配置
BACKTEST_CONFIG = {
    "commission_rate": 0.0,  # 手續費率(選擇權買方無手續費概念,僅權利金)
    "slippage": 0.0,  # 滑價
    "position_sizing": "fixed",  # 部位大小策略: fixed/kelly
    "max_positions": 2  # 最大同時持倉數
}

# === 以下為模型直接使用的參數 ===

# XGBoost 方向性預測模型參數
DIRECTION_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softmax',
    'num_class': 3,  # 0: 跌, 1: 盤整, 2: 漲
    'random_state': 42,
    'n_jobs': -1
}

# LSTM 波動率預測模型參數
VOLATILITY_MODEL_PARAMS = {
    'lookback': 30,  # 回看天數
    'hv_window': 20,  # 歷史波動率計算窗口
    'epochs': 50,
    'batch_size': 32
}

# 特徵欄位
FEATURE_COLUMNS = [
    # 價格相關
    'close', 'open', 'high', 'low',
    # 成交量
    'volume',
    # 技術指標
    'rsi', 'macd', 'macd_signal', 'macd_diff',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'atr', 'adx',
    # 移動平均
    'sma_5', 'sma_20', 'sma_60',
    'ema_12', 'ema_26',
    # 動量指標
    'momentum', 'roc',
    # 波動率
    'historical_volatility'
]
