"""
購入點信號生成器
偵測模型預測買點 + 風險檢查 + Discord 通知
"""
import sys
from pathlib import Path
from datetime import datetime

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from loguru import logger

from src.data.database import Database
from src.features.technical import add_all_technical_indicators
from src.utils.risk_monitor import RiskMonitor
from src.notification.discord_bot import DiscordNotifier

# ===== 設定參數 =====
CONFIDENCE_THRESHOLD = 0.70   # 信心度閾值
STOP_LOSS_PCT = 0.03          # 停損 3%
TAKE_PROFIT_PCT = 0.30        # 停利 30%
ATR_ANOMALY_RATIO = 2.0       # ATR 異常倍數
LOOKBACK_DAYS = 60            # 風險檢查回看天數
STRIKE_INTERVAL = 50          # 週選履約價間距 (點)


def load_models():
    """載入所有模型"""
    models = {}
    
    # XGBoost (主模型)
    xgb_path = Path("models/direction_model.pkl")
    if xgb_path.exists():
        models['xgboost'] = joblib.load(xgb_path)
        logger.info("[OK] XGBoost 模型已載入")
    else:
        logger.error("XGBoost 模型不存在!")
        return None
    
    # LightGBM
    lgbm_path = Path("models/lgbm_model.pkl")
    if lgbm_path.exists():
        models['lightgbm'] = joblib.load(lgbm_path)
        logger.info("[OK] LightGBM 模型已載入")
    
    # Meta Learner
    meta_path = Path("models/meta_learner.pkl")
    if meta_path.exists():
        models['meta_learner'] = joblib.load(meta_path)
        logger.info("[OK] Meta Learner 已載入")
    
    # 特徵欄位
    cols_path = Path("models/feature_cols.json")
    if cols_path.exists():
        with open(cols_path, 'r') as f:
            models['feature_cols'] = json.load(f)
        logger.info(f"[OK] 特徵欄位: {len(models['feature_cols'])} 個")
    else:
        logger.error("feature_cols.json 不存在!")
        return None
    
    return models


def get_latest_data():
    """取得最新市場資料 (含技術指標)"""
    with Database() as db:
        df = db.get_futures_data()
    
    if df.empty or len(df) < 30:
        logger.error(f"資料不足: {len(df)} 筆")
        return None
    
    # 先過濾異常資料
    df = df[(df['close'] > 0) & (df['volume'] > 0)]
    
    # 每日只保留主力合約 (交易量最大者) — 必須在技術指標計算之前
    df = df.loc[df.groupby('date')['volume'].idxmax()]
    df = df.sort_values('date').reset_index(drop=True)
    
    # 在乾淨的主力合約資料上計算技術指標
    df = add_all_technical_indicators(df)
    
    # 處理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    logger.info(f"[OK] 資料載入: {len(df)} 筆, 最新日期: {df['date'].max()}")
    return df


def predict_direction(models, df):
    """使用模型預測方向"""
    feature_cols = models['feature_cols']
    latest = df.iloc[-1:]
    
    X = latest[feature_cols].fillna(0)
    
    # XGBoost 預測
    xgb_model = models['xgboost']
    xgb_pred = xgb_model.predict(X)[0]
    xgb_proba = xgb_model.predict_proba(X)[0]
    xgb_confidence = float(np.max(xgb_proba))
    
    # 類別對應: 0=跌, 1=盤整, 2=漲
    direction_map = {0: 'PUT', 1: 'HOLD', 2: 'CALL'}
    xgb_direction = direction_map.get(int(xgb_pred), 'HOLD')
    
    result = {
        'xgb_direction': xgb_direction,
        'xgb_confidence': xgb_confidence,
        'xgb_proba': xgb_proba.tolist(),
    }
    
    # LightGBM 預測 (若有)
    lgbm_proba_full = None
    if 'lightgbm' in models:
        lgbm_model = models['lightgbm']
        lgbm_pred = lgbm_model.predict(X)[0]
        lgbm_proba = lgbm_model.predict_proba(X)[0]
        lgbm_confidence = float(np.max(lgbm_proba))
        lgbm_direction = direction_map.get(int(lgbm_pred), 'HOLD')
        
        # 確保 3 個類別的機率
        lgbm_proba_full = np.zeros(3)
        for i, p in enumerate(lgbm_proba):
            if i < 3:
                lgbm_proba_full[i] = p
        
        result['lgbm_direction'] = lgbm_direction
        result['lgbm_confidence'] = lgbm_confidence
        result['models_agree'] = (xgb_direction == lgbm_direction)
    else:
        result['lgbm_direction'] = None
        result['lgbm_confidence'] = 0
        result['models_agree'] = True
    
    # 確保 XGB proba 也有 3 個類別
    xgb_proba_full = np.zeros(3)
    for i, p in enumerate(xgb_proba):
        if i < 3:
            xgb_proba_full[i] = p
    
    # Meta Learner (若有)
    if 'meta_learner' in models and lgbm_proba_full is not None:
        meta = models['meta_learner']
        # 組裝 6 維特徵: XGB(3) + LGBM(3)
        meta_features = np.concatenate([xgb_proba_full, lgbm_proba_full]).reshape(1, -1)
        meta_pred = meta.predict(meta_features)[0]
        meta_proba = meta.predict_proba(meta_features)[0]
        
        result['final_direction'] = direction_map.get(int(meta_pred), 'HOLD')
        result['final_confidence'] = float(np.max(meta_proba))
    else:
        result['final_direction'] = xgb_direction
        result['final_confidence'] = xgb_confidence
    
    return result


def check_risks(df):
    """風險檢查"""
    risks = {
        'warnings': [],
        'risk_level': 'LOW',
        'atr_status': 'NORMAL',
        'should_trade': True
    }
    
    latest = df.iloc[-1]
    recent = df.tail(LOOKBACK_DAYS)
    
    # 1. ATR 異常偵測
    if 'atr' in df.columns:
        avg_atr = recent['atr'].mean()
        current_atr = latest['atr']
        
        if avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            
            if atr_ratio > ATR_ANOMALY_RATIO * 1.5:
                risks['atr_status'] = 'EXTREME'
                risks['risk_level'] = 'EXTREME'
                risks['warnings'].append(
                    f"[!!] ATR 極端異常! 當前 {current_atr:.0f} 為平均 {avg_atr:.0f} 的 {atr_ratio:.1f} 倍"
                )
                risks['should_trade'] = False
            elif atr_ratio > ATR_ANOMALY_RATIO:
                risks['atr_status'] = 'HIGH'
                risks['risk_level'] = 'HIGH'
                risks['warnings'].append(
                    f"[!] ATR 偏高: 當前 {current_atr:.0f} 為平均 {avg_atr:.0f} 的 {atr_ratio:.1f} 倍"
                )
    
    # 2. 價格跳空偵測
    if len(df) >= 2:
        prev_close = df.iloc[-2]['close']
        today_open = latest['open']
        gap_pct = abs(today_open - prev_close) / prev_close
        
        if gap_pct > 0.02:  # 跳空超過 2%
            risks['risk_level'] = 'HIGH'
            risks['warnings'].append(
                f"[!] 跳空 {gap_pct:.1%}: 前收 {prev_close:.0f} -> 今開 {today_open:.0f}"
            )
    
    # 3. RSI 極端
    if 'rsi' in df.columns:
        rsi = latest['rsi']
        if rsi > 80:
            risks['warnings'].append(f"[!] RSI 超買: {rsi:.1f}")
        elif rsi < 20:
            risks['warnings'].append(f"[!] RSI 超賣: {rsi:.1f}")
    
    # 4. 連續同向判斷 (近 5 日)
    if len(df) >= 5:
        last_5_returns = df.tail(5)['close'].pct_change().dropna()
        if (last_5_returns > 0).all():
            risks['warnings'].append("[!] 連漲 5 日, 注意回調風險")
        elif (last_5_returns < 0).all():
            risks['warnings'].append("[!] 連跌 5 日, 可能有超賣反彈")
    
    return risks


def recommend_strikes(current_price, direction, confidence, df):
    """
    推薦週選履約價
    
    Args:
        current_price: 當前台指期價格
        direction: CALL 或 PUT
        confidence: 模型信心度
        df: 歷史資料 (for ATR/波動率計算)
    
    Returns:
        dict with strike recommendations for 3 levels
    """
    # 計算歷史波動幅度
    recent = df.tail(60)
    daily_returns = recent['close'].pct_change().dropna()
    avg_daily_move = daily_returns.abs().mean()  # 平均每日絕對波動
    std_daily_move = daily_returns.std()
    
    # 週選持有期間 (5 個交易日)
    holding_days = 5
    expected_move_pct = avg_daily_move * np.sqrt(holding_days)  # 預期一週波動
    expected_move_pts = current_price * expected_move_pct
    
    # ATR 基磎移動脚悳
    if 'atr' in df.columns:
        atr = df.iloc[-1]['atr']
        weekly_atr = atr * np.sqrt(holding_days)
    else:
        weekly_atr = expected_move_pts
    
    # 取較大值作為參考
    ref_move = max(expected_move_pts, weekly_atr)
    
    # 履約價對齊到 STRIKE_INTERVAL 的倍數
    atm_strike = round(current_price / STRIKE_INTERVAL) * STRIKE_INTERVAL
    
    # 三檔履約價推薦
    strikes = {}
    
    if direction == 'CALL':
        # 保守: 價內 1~2 檔 (ITM/ATM), 成本高但勝率高
        conservative_strike = atm_strike
        # 穩健: 價外 2~4 檔, 中等成本
        moderate_offset = round(ref_move * 0.3 / STRIKE_INTERVAL) * STRIKE_INTERVAL
        moderate_offset = max(moderate_offset, STRIKE_INTERVAL * 2)  # 至少 100 點
        moderate_strike = atm_strike + moderate_offset
        # 積極: 價外 5~8 檔, 低成本高槓桿 (樂透式)
        aggressive_offset = round(ref_move * 0.7 / STRIKE_INTERVAL) * STRIKE_INTERVAL
        aggressive_offset = max(aggressive_offset, STRIKE_INTERVAL * 5)  # 至少 250 點
        aggressive_strike = atm_strike + aggressive_offset
    else:  # PUT
        conservative_strike = atm_strike
        moderate_offset = round(ref_move * 0.3 / STRIKE_INTERVAL) * STRIKE_INTERVAL
        moderate_offset = max(moderate_offset, STRIKE_INTERVAL * 2)
        moderate_strike = atm_strike - moderate_offset
        aggressive_offset = round(ref_move * 0.7 / STRIKE_INTERVAL) * STRIKE_INTERVAL
        aggressive_offset = max(aggressive_offset, STRIKE_INTERVAL * 5)
        aggressive_strike = atm_strike - aggressive_offset
    
    # 距離計算
    for level, strike, desc in [
        ('conservative', conservative_strike, '保守 (ATM/價內)'),
        ('moderate', moderate_strike, '穩健 (OTM)'),
        ('aggressive', aggressive_strike, '積極 (Deep OTM)'),
    ]:
        distance = abs(strike - current_price)
        distance_pct = distance / current_price * 100
        
        # 粗略估算成為價內機率 (based on historical move distribution)
        if direction == 'CALL':
            # CALL 的目標: 價格要漲超過 strike
            target_move_pct = (strike - current_price) / current_price
        else:
            # PUT 的目標: 價格要跌破 strike
            target_move_pct = (current_price - strike) / current_price
        
        # 用歷史統計估算機率
        weekly_returns = daily_returns.rolling(holding_days).sum().dropna()
        if direction == 'CALL':
            hit_rate = (weekly_returns > target_move_pct).mean()
        else:
            hit_rate = (weekly_returns < -target_move_pct).mean()
        
        strikes[level] = {
            'strike': int(strike),
            'description': desc,
            'distance': int(distance),
            'distance_pct': round(distance_pct, 2),
            'est_hit_rate': round(float(hit_rate) * 100, 1),
        }
    
    return {
        'strikes': strikes,
        'atm_strike': int(atm_strike),
        'ref_move': round(ref_move, 0),
        'weekly_atr': round(weekly_atr, 0),
        'avg_daily_move_pct': round(avg_daily_move * 100, 2),
    }


def determine_signal(prediction, risks):
    """判定信號 - 所有結果都通知，標註信心度讓用戶自行判斷"""
    direction = prediction['final_direction']
    confidence = prediction['final_confidence']
    models_agree = prediction['models_agree']
    
    signal = {
        'action': 'HOLD',
        'direction': direction,
        'confidence': confidence,
        'models_agree': models_agree,
        'risk_level': risks['risk_level'],
        'should_notify': True,  # 永遠通知
        'reason': ''
    }
    
    # 極端風險
    if not risks['should_trade']:
        signal['reason'] = '極端風險, 建議暫停交易'
        signal['action'] = 'WARNING'
        return signal
    
    # 模型不一致提醒
    if not models_agree:
        risks['warnings'].append("[!] XGBoost 與 LightGBM 預測不一致")
    
    if direction == 'HOLD':
        signal['action'] = 'HOLD'
        signal['reason'] = f'模型預測盤整 (信心度 {confidence:.1%})'
    else:
        signal['action'] = f'BUY_{direction}'
        agree_str = '一致' if models_agree else '不一致'
        signal['reason'] = f'信心度 {confidence:.1%}, 模型{agree_str}'
    
    return signal


def send_discord_signal(signal, prediction, risks, market_data, strike_info=None):
    """發送 Discord 通知"""
    try:
        notifier = DiscordNotifier()
        
        # 決定顏色和文字
        if signal['action'] == 'WARNING':
            color = 0xFF0000
            title = "[!!] 極端風險警報"
            action_text = "暫停交易 - 市場波動異常"
        elif signal['action'] == 'BUY_CALL':
            color = 0x00FF00
            title = "[SIGNAL] 選擇權買入信號 - CALL"
            action_text = "Buy Call (看漲)"
        elif signal['action'] == 'BUY_PUT':
            color = 0xFF6600
            title = "[SIGNAL] 週選買入信號 - PUT"
            action_text = "Buy Put (看跌)"
        else:  # HOLD
            color = 0xFFFF00
            title = "[INFO] 每日市場分析"
            action_text = f"觀望 - {signal['reason']}"
        
        # 風險等級標示
        risk_icons = {
            'LOW': '[OK] 低',
            'HIGH': '[!] 高',
            'EXTREME': '[!!] 極高'
        }
        risk_text = risk_icons.get(signal['risk_level'], '[?] 未知')
        
        # 模型一致性
        agree_text = "[OK] 一致" if prediction['models_agree'] else "[!] 不一致"
        model_detail = (
            f"XGBoost: {prediction['xgb_direction']} ({prediction['xgb_confidence']:.1%})\n"
            f"LightGBM: {prediction.get('lgbm_direction', 'N/A')} ({prediction.get('lgbm_confidence', 0):.1%})\n"
            f"Stacking: {prediction['final_direction']} ({prediction['final_confidence']:.1%})"
        )
        
        # 停損停利建議
        close_price = market_data.get('close', 0)
        
        embed = {
            "title": title,
            "description": f"**策略: {action_text}**",
            "color": color,
            "fields": [
                {
                    "name": "[Market] 台指期現況",
                    "value": (
                        f"收盤: {close_price:,.0f}\n"
                        f"漲跌: {market_data.get('change', 0):+.0f} ({market_data.get('change_pct', 0):+.2f}%)\n"
                        f"成交量: {market_data.get('volume', 0):,.0f}"
                    ),
                    "inline": True
                },
                {
                    "name": "[AI] 模型預測",
                    "value": model_detail,
                    "inline": True
                },
                {
                    "name": "[CHECK] 一致性 / 信心度",
                    "value": f"{agree_text}\n信心度: {signal['confidence']:.1%}",
                    "inline": True
                },
            ],
            "footer": {
                "text": f"Signal Generator | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        # 履約價建議 (核心欄位)
        if strike_info and signal['action'] != 'WARNING':
            strikes = strike_info['strikes']
            direction = signal['direction']
            cons = strikes['conservative']
            mod = strikes['moderate']
            agg = strikes['aggressive']
            
            embed["fields"].extend([
                {
                    "name": f"[A] 保守 {direction} {cons['strike']:,}",
                    "value": (
                        f"距離: {cons['distance']:,} 點 ({cons['distance_pct']:.1f}%)\n"
                        f"歷史命中率: {cons['est_hit_rate']:.0f}%\n"
                        f"特性: 成本高/勝率高"
                    ),
                    "inline": True
                },
                {
                    "name": f"[B] 穩健 {direction} {mod['strike']:,}",
                    "value": (
                        f"距離: {mod['distance']:,} 點 ({mod['distance_pct']:.1f}%)\n"
                        f"歷史命中率: {mod['est_hit_rate']:.0f}%\n"
                        f"特性: 中等成本/中等勝率"
                    ),
                    "inline": True
                },
                {
                    "name": f"[C] 積極 {direction} {agg['strike']:,}",
                    "value": (
                        f"距離: {agg['distance']:,} 點 ({agg['distance_pct']:.1f}%)\n"
                        f"歷史命中率: {agg['est_hit_rate']:.0f}%\n"
                        f"特性: 低成本/高槓桿"
                    ),
                    "inline": True
                },
                {
                    "name": "[VOL] 波動參考",
                    "value": (
                        f"ATM: {strike_info['atm_strike']:,}\n"
                        f"週化 ATR: {strike_info['weekly_atr']:.0f} 點\n"
                        f"每日平均波動: {strike_info['avg_daily_move_pct']}%"
                    ),
                    "inline": True
                },
            ])
        
        # 風險 + 操作建議
        embed["fields"].append({
            "name": "[RISK] 風險等級",
            "value": risk_text,
            "inline": True
        })
        
        if signal['action'] != 'WARNING':
            embed["fields"].append({
                "name": "[SL/TP] 週選操作建議",
                "value": (
                    "停損: 最多虧損 50% 權利金\n"
                    "停利: 獲利 100~300% 分批出場\n"
                    "持有: 週選到期前 1~2 天"
                ),
                "inline": True
            })
        
        # 風險警告
        if risks['warnings']:
            embed["fields"].append({
                "name": "[WARN] 風險警告",
                "value": "\n".join(risks['warnings']),
                "inline": False
            })
        
        # 免責聲明
        embed["fields"].append({
            "name": "[NOTE] 免責聲明",
            "value": "樂透式週選交易, 僅供參考。請以可承受虐損金額操作。",
            "inline": False
        })
        
        notifier.send_message("", embeds=[embed])
        logger.info("[OK] Discord 信號通知已發送")
        return True
        
    except Exception as e:
        logger.error(f"Discord 通知發送失敗: {e}")
        return False


def log_signal(signal, prediction, risks, market_data):
    """記錄信號到 CSV"""
    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "signals_log.csv"
    
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': market_data.get('date', ''),
        'close': market_data.get('close', 0),
        'action': signal['action'],
        'direction': signal['direction'],
        'confidence': signal['confidence'],
        'models_agree': signal['models_agree'],
        'risk_level': signal['risk_level'],
        'xgb_direction': prediction['xgb_direction'],
        'xgb_confidence': prediction['xgb_confidence'],
        'lgbm_direction': prediction.get('lgbm_direction', ''),
        'lgbm_confidence': prediction.get('lgbm_confidence', 0),
        'atr_status': risks['atr_status'],
        'warnings': '; '.join(risks['warnings']),
        'reason': signal['reason'],
    }
    
    record_df = pd.DataFrame([record])
    
    if log_file.exists():
        record_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        record_df.to_csv(log_file, index=False)
    
    logger.info(f"[OK] 信號已記錄: {log_file}")


def generate_signal():
    """主流程: 生成購入點信號"""
    logger.info("=" * 60)
    logger.info("=== 購入點信號生成器 ===")
    logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 1. 載入模型
    models = load_models()
    if models is None:
        logger.error("模型載入失敗, 中止")
        return None
    
    # 2. 取得最新資料
    df = get_latest_data()
    if df is None:
        logger.error("資料取得失敗, 中止")
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    
    market_data = {
        'date': str(latest.get('date', '')),
        'close': float(latest['close']),
        'open': float(latest['open']),
        'high': float(latest['high']),
        'low': float(latest['low']),
        'volume': float(latest['volume']),
        'change': float(latest['close'] - prev['close']),
        'change_pct': float((latest['close'] - prev['close']) / prev['close'] * 100),
    }
    
    logger.info(f"\n[Market] 台指期: {market_data['close']:,.0f} "
                f"({market_data['change']:+.0f}, {market_data['change_pct']:+.2f}%)")
    
    # 3. 模型預測
    prediction = predict_direction(models, df)
    logger.info(f"\n[AI] 預測結果:")
    logger.info(f"  XGBoost:  {prediction['xgb_direction']} ({prediction['xgb_confidence']:.1%})")
    if prediction.get('lgbm_direction'):
        logger.info(f"  LightGBM: {prediction['lgbm_direction']} ({prediction['lgbm_confidence']:.1%})")
    logger.info(f"  Stacking: {prediction['final_direction']} ({prediction['final_confidence']:.1%})")
    logger.info(f"  模型一致: {'YES' if prediction['models_agree'] else 'NO'}")
    
    # 4. 風險檢查
    risks = check_risks(df)
    logger.info(f"\n[RISK] 風險檢查:")
    logger.info(f"  風險等級: {risks['risk_level']}")
    logger.info(f"  ATR 狀態: {risks['atr_status']}")
    if risks['warnings']:
        for w in risks['warnings']:
            logger.info(f"  {w}")
    else:
        logger.info("  無特殊風險")
    
    # 5. 信號判定
    signal = determine_signal(prediction, risks)
    logger.info(f"\n[SIGNAL] 信號判定:")
    logger.info(f"  動作: {signal['action']}")
    logger.info(f"  理由: {signal['reason']}")
    logger.info(f"  發送通知: {'YES' if signal['should_notify'] else 'NO'}")
    
    # 6. 履約價建議 (所有情況都計算)
    strike_info = None
    if signal['action'] in ('BUY_CALL', 'BUY_PUT'):
        direction = signal['direction']
        strike_info = recommend_strikes(
            market_data['close'], direction, signal['confidence'], df
        )
    else:
        # HOLD/WARNING 也計算參考履約價 (以 CALL 為基礎)
        strike_info = recommend_strikes(
            market_data['close'], 'CALL', 0, df
        )
    
    logger.info(f"\n[STRIKE] 週選參考:")
    logger.info(f"  ATM: {strike_info['atm_strike']:,}")
    logger.info(f"  週化 ATR: {strike_info['weekly_atr']:.0f} 點")
    logger.info(f"  每日平均波動: {strike_info['avg_daily_move_pct']}%")
    for level, info in strike_info['strikes'].items():
        logger.info(f"  [{level}] Strike {info['strike']:,} | "
                   f"距離 {info['distance']:,} ({info['distance_pct']}%) | "
                   f"命中率 {info['est_hit_rate']:.0f}%")
    
    # 7. 記錄信號
    log_signal(signal, prediction, risks, market_data)
    
    # 8. 發送 Discord 通知 (若需要)
    if signal['should_notify']:
        logger.info("\n[NOTIFY] 發送 Discord 通知...")
        send_discord_signal(signal, prediction, risks, market_data, strike_info)
    else:
        logger.info("\n[SKIP] 未觸發通知條件")
    
    return signal


if __name__ == "__main__":
    try:
        signal = generate_signal()
        
        print("\n" + "=" * 60)
        if signal and signal['should_notify']:
            print(f"[SIGNAL] {signal['action']} | "
                  f"信心度: {signal['confidence']:.1%} | "
                  f"風險: {signal['risk_level']}")
        elif signal:
            print(f"[HOLD] 觀望 | 原因: {signal['reason']}")
        else:
            print("[ERROR] 信號生成失敗")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"信號生成異常: {e}")
        import traceback
        traceback.print_exc()
