"""
Step 2: 模型訓練 (XGBoost + LightGBM Stacking Ensemble)
"""
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import joblib
import json

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM 未安裝, 將僅使用 XGBoost")

# 排除原始價格欄位 (防止價格洩漏)
RAW_PRICE_COLS = ['open', 'high', 'low', 'close', 'volume',
                  'spread', 'spread_per', 'atr']
EXCLUDE_COLS = ['date', 'futures_id', 'contract_date', 'trading_session'] + RAW_PRICE_COLS


def get_feature_cols(df):
    """取得有效特徵欄位 (排除原始價格和標籤)"""
    return [col for col in df.columns 
            if col not in EXCLUDE_COLS
            and not col.startswith('return_') 
            and not col.startswith('label_')]


def train_direction_model():
    """訓練方向性預測模型 (XGBoost + LightGBM Stacking Ensemble)"""
    logger.info("=== 訓練方向性預測模型 (Stacking Ensemble) ===")
    
    # 載入訓練資料
    train_df = pd.read_csv("data/train/train_labeled.csv")
    
    # 使用安全特徵欄位
    feature_cols = get_feature_cols(train_df)
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['label_5d'].fillna(0).astype(int)  # 使用 5 天預測
    
    # 移除無效標籤
    valid_mask = ~y_train.isna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    logger.info(f"訓練資料: {len(X_train)} 筆")
    logger.info(f"特徵數量: {len(feature_cols)} (已排除原始價格)")
    logger.info(f"標籤分布: 漲={sum(y_train==1)}, 盤={sum(y_train==0)}, 跌={sum(y_train==-1)}")
    
    # 排除的欄位提示
    excluded_in_data = [c for c in RAW_PRICE_COLS if c in train_df.columns]
    if excluded_in_data:
        logger.info(f"已排除原始價格欄位: {excluded_in_data}")
    
    # 時序交叉驗證 (防止未來資料洩漏)
    n_splits = min(5, max(2, len(X_train) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 資料太少,使用簡化訓練
    if len(X_train) < 10:
        logger.warning(f"[WARN] 訓練資料不足 ({len(X_train)} 筆),使用簡化訓練模式")
        
        best_model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            reg_alpha=1.0,
            reg_lambda=5.0,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        best_model.fit(X_train, y_train + 1)
        
        best_params = {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'min_child_weight': 3,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'gamma': 0.1
        }
        cv_score = 0.0
        
    else:
        # === XGBoost 訓練 (增強正則化) ===
        logger.info("訓練 XGBoost (增強正則化)...")
        
        # 使用合理的參數範圍做隨機搜尋
        from sklearn.model_selection import RandomizedSearchCV
        
        param_distributions = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [3, 5, 7],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [0.1, 1.0, 5.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'gamma': [0, 0.1, 0.3]
        }
        
        base_xgb = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        logger.info(f"執行隨機搜尋 (TimeSeriesSplit, {n_splits} 折)...")
        random_search = RandomizedSearchCV(
            base_xgb,
            param_distributions,
            n_iter=30,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train + 1)
        best_xgb = random_search.best_estimator_
        best_params = random_search.best_params_
        logger.info(f"XGBoost 最佳參數: {best_params}")
        
        # 時序交叉驗證
        cv_scores = cross_val_score(best_xgb, X_train, y_train + 1, cv=tscv, scoring='accuracy')
        cv_score = float(cv_scores.mean())
        logger.info(f"XGBoost 交叉驗證準確率: {cv_score:.4f} (+/- {cv_scores.std():.4f})")
        
        # === LightGBM 訓練 ===
        if HAS_LGBM:
            logger.info("訓練 LightGBM...")
            
            lgbm_model = LGBMClassifier(
                objective='multiclass',
                num_class=3,
                max_depth=best_params.get('max_depth', 4),
                learning_rate=0.05,
                n_estimators=200,
                min_child_weight=best_params.get('min_child_weight', 5),
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=42,
                verbose=-1
            )
            
            lgbm_model.fit(X_train, y_train + 1)
            
            lgbm_cv_scores = cross_val_score(lgbm_model, X_train, y_train + 1, cv=tscv, scoring='accuracy')
            logger.info(f"LightGBM 交叉驗證準確率: {lgbm_cv_scores.mean():.4f} (+/- {lgbm_cv_scores.std():.4f})")
            
            # === Stacking Ensemble ===
            logger.info("建立 Stacking Ensemble...")
            
            # 生成 meta 特徵 (使用 out-of-fold 預測)
            meta_train = np.zeros((len(X_train), 6))  # 3*2 classes probabilities
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = (y_train.iloc[train_idx] + 1)
                X_fold_val = X_train.iloc[val_idx]
                
                # XGBoost fold
                xgb_fold = XGBClassifier(**{**best_params,
                    'objective': 'multi:softmax',
                    'num_class': 3,
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                })
                xgb_fold.fit(X_fold_train, y_fold_train)
                xgb_proba = xgb_fold.predict_proba(X_fold_val)
                
                # LightGBM fold
                lgbm_fold = LGBMClassifier(
                    objective='multiclass', num_class=3,
                    max_depth=best_params.get('max_depth', 4),
                    learning_rate=0.05, n_estimators=200,
                    random_state=42, verbose=-1
                )
                lgbm_fold.fit(X_fold_train, y_fold_train)
                lgbm_proba = lgbm_fold.predict_proba(X_fold_val)
                
                meta_train[val_idx, :3] = xgb_proba
                meta_train[val_idx, 3:] = lgbm_proba
            
            # 訓練 Meta Learner
            meta_learner = LogisticRegression(
                max_iter=1000, random_state=42, C=0.1
            )
            meta_learner.fit(meta_train, y_train + 1)
            
            # Meta CV 評估
            meta_cv_scores = cross_val_score(
                meta_learner, meta_train, y_train + 1, cv=tscv, scoring='accuracy'
            )
            logger.info(f"Stacking Ensemble 交叉驗證: {meta_cv_scores.mean():.4f} (+/- {meta_cv_scores.std():.4f})")
            
            # 儲存 LightGBM 和 Meta Learner
            models_dir = Path("models")
            joblib.dump(lgbm_model, models_dir / "lgbm_model.pkl")
            joblib.dump(meta_learner, models_dir / "meta_learner.pkl")
            logger.info("[OK] LightGBM 和 Meta Learner 已儲存")
            
            cv_score = float(meta_cv_scores.mean())
        
        best_model = best_xgb
    
    # 訓練最終模型
    best_model.fit(X_train, y_train + 1)
    
    # 訓練集評估
    y_pred = best_model.predict(X_train)
    y_true = y_train + 1
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    logger.info(f"\n訓練集評估:")
    logger.info(f"  準確率: {accuracy:.4f}")
    logger.info(f"  精確率: {precision:.4f}")
    logger.info(f"  召回率: {recall:.4f}")
    logger.info(f"  F1分數: {f1:.4f}")
    logger.info(f"  交叉驗證: {cv_score:.4f}")
    
    # 過擬合監控
    overfit_gap = accuracy - cv_score
    if overfit_gap > 0.2:
        logger.warning(f"[WARN] 訓練/CV 差距 {overfit_gap:.2%}, 仍有過擬合風險")
    else:
        logger.info(f"[OK] 訓練/CV 差距 {overfit_gap:.2%}, 泛化合理")
    
    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n混淆矩陣:")
    logger.info(f"\n{cm}")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 10 重要特徵:")
    print(feature_importance.head(10))
    
    # 檢查是否有價格洩漏
    leak_features = [f for f in feature_importance.head(5)['feature'] if f in RAW_PRICE_COLS]
    if leak_features:
        logger.warning(f"[WARN] 發現價格洩漏特徵: {leak_features}")
    else:
        logger.info("[OK] 特徵無價格洩漏")
    
    # 儲存模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "direction_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"\n[OK] 模型已儲存: {model_path}")
    
    # 儲存特徵欄位列表 (供回測使用)
    with open(models_dir / "feature_cols.json", 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    
    # 儲存元資料
    metadata = {
        'model_type': 'XGBClassifier + Stacking Ensemble' if HAS_LGBM else 'XGBClassifier',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(X_train),
        'num_features': len(feature_cols),
        'best_params': best_params,
        'cv_type': 'TimeSeriesSplit',
        'cv_folds': n_splits,
        'cv_score': cv_score,
        'train_accuracy': float(accuracy),
        'train_precision': float(precision),
        'train_recall': float(recall),
        'train_f1': float(f1),
        'overfit_gap': float(overfit_gap),
        'has_lgbm': HAS_LGBM,
        'excluded_price_cols': excluded_in_data,
        'feature_importance': feature_importance.head(20).to_dict('records')
    }
    
    with open(models_dir / "direction_model_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return best_model, metadata

def train_volatility_model():
    """訓練波動率預測模型 (簡化版)"""
    logger.info("\n=== 訓練波動率預測模型 ===")
    
    # 載入訓練資料
    train_df = pd.read_csv("data/train/train_labeled.csv")
    
    # 計算歷史波動率
    if 'historical_volatility' not in train_df.columns:
        train_df['historical_volatility'] = train_df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 使用簡單的移動平均作為預測
    train_df['predicted_volatility'] = train_df['historical_volatility'].rolling(5).mean()
    
    # 計算 MAE
    valid_mask = ~train_df['predicted_volatility'].isna() & ~train_df['historical_volatility'].isna()
    mae = np.abs(
        train_df.loc[valid_mask, 'predicted_volatility'] - 
        train_df.loc[valid_mask, 'historical_volatility']
    ).mean()
    
    logger.info(f"訓練集 MAE: {mae:.6f}")
    
    # 儲存簡單元資料
    metadata = {
        'model_type': 'SimpleMovingAverage',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_mae': float(mae),
        'note': '使用 5 日移動平均預測波動率'
    }
    
    models_dir = Path("models")
    with open(models_dir / "volatility_model_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info("[OK] 波動率模型元資料已儲存")
    
    return metadata

if __name__ == "__main__":
    logger.info("=== 開始模型訓練 ===\n")
    
    try:
        # 訓練方向性模型
        direction_model, direction_metadata = train_direction_model()
        
        # 訓練波動率模型
        volatility_metadata = train_volatility_model()
        
        print("\n" + "="*60)
        print("[SUCCESS] 模型訓練完成!")
        print("="*60)
        print(f"\n方向性模型:")
        print(f"  類型: {direction_metadata['model_type']}")
        print(f"  訓練準確率: {direction_metadata['train_accuracy']:.2%}")
        print(f"  交叉驗證 (TimeSeriesSplit): {direction_metadata['cv_score']:.2%}")
        print(f"  過擬合差距: {direction_metadata['overfit_gap']:.2%}")
        print(f"  特徵數: {direction_metadata['num_features']}")
        print(f"\n波動率模型 MAE: {volatility_metadata['train_mae']:.6f}")
        print("\n下一步: python scripts/3_optimize_strategy.py")
        
    except Exception as e:
        logger.error(f"訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("[ERROR] 模型訓練失敗!")
        print("="*60)

