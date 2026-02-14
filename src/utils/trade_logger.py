"""
交易日誌系統
記錄所有交易細節與績效指標
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger


class TradeLogger:
    """交易日誌記錄器"""
    
    def __init__(self, log_dir: str = "logs/trades"):
        """
        初始化交易日誌
        
        Args:
            log_dir: 日誌目錄
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_file = self.log_dir / f"session_{self.current_session}.json"
        self.trades: List[Dict] = []
        
        logger.info(f"交易日誌初始化: {self.session_file}")
    
    def log_prediction(
        self,
        date: datetime,
        prediction: Dict,
        market_data: Dict
    ) -> None:
        """
        記錄預測結果
        
        Args:
            date: 預測日期
            prediction: 預測結果
            market_data: 市場資料
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'date': date.isoformat(),
            'type': 'PREDICTION',
            'prediction': prediction,
            'market_data': market_data
        }
        
        self._save_log(log_entry)
        logger.info(f"記錄預測: {date.date()} - {prediction.get('action', 'UNKNOWN')}")
    
    def log_trade_entry(
        self,
        date: datetime,
        action: str,  # 'BUY_CALL' or 'BUY_PUT'
        strike: float,
        premium: float,
        confidence: float,
        position_size: float,
        reason: str
    ) -> str:
        """
        記錄進場交易
        
        Args:
            date: 交易日期
            action: 交易動作
            strike: 履約價
            premium: 權利金
            confidence: 信心度
            position_size: 部位大小
            reason: 進場理由
        
        Returns:
            交易 ID
        """
        trade_id = f"{date.strftime('%Y%m%d')}_{len(self.trades)+1:03d}"
        
        trade = {
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'entry_date': date.isoformat(),
            'action': action,
            'strike': strike,
            'entry_premium': premium,
            'confidence': confidence,
            'position_size': position_size,
            'reason': reason,
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        self._save_log({'type': 'ENTRY', **trade})
        
        logger.info(f"記錄進場: {trade_id} - {action} @ {premium}")
        
        return trade_id
    
    def log_trade_exit(
        self,
        trade_id: str,
        exit_date: datetime,
        exit_premium: float,
        pnl: float,
        return_pct: float,
        reason: str
    ) -> None:
        """
        記錄出場交易
        
        Args:
            trade_id: 交易 ID
            exit_date: 出場日期
            exit_premium: 出場權利金
            pnl: 損益
            return_pct: 報酬率
            reason: 出場理由
        """
        # 找到對應的交易
        trade = next((t for t in self.trades if t['trade_id'] == trade_id), None)
        
        if trade is None:
            logger.error(f"找不到交易 ID: {trade_id}")
            return
        
        # 更新交易資訊
        trade.update({
            'exit_date': exit_date.isoformat(),
            'exit_premium': exit_premium,
            'pnl': pnl,
            'return_pct': return_pct,
            'exit_reason': reason,
            'status': 'CLOSED',
            'holding_days': (exit_date - datetime.fromisoformat(trade['entry_date'])).days
        })
        
        self._save_log({'type': 'EXIT', **trade})
        
        logger.info(f"記錄出場: {trade_id} - 損益 {pnl:+,.0f} ({return_pct:+.2%})")
    
    def log_risk_event(
        self,
        event_type: str,  # 'VOLATILITY_ANOMALY', 'CONSECUTIVE_LOSS', 'EMERGENCY_STOP'
        severity: str,  # 'INFO', 'WARNING', 'CRITICAL'
        details: Dict
    ) -> None:
        """
        記錄風險事件
        
        Args:
            event_type: 事件類型
            severity: 嚴重程度
            details: 事件詳情
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'RISK_EVENT',
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        self._save_log(log_entry)
        
        log_func = {
            'INFO': logger.info,
            'WARNING': logger.warning,
            'CRITICAL': logger.error
        }.get(severity, logger.info)
        
        log_func(f"風險事件: {event_type} - {details}")
    
    def _save_log(self, log_entry: Dict) -> None:
        """儲存日誌到檔案"""
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_session_summary(self) -> Dict:
        """
        取得本次交易摘要
        
        Returns:
            交易摘要統計
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'open_trades': 0,
                'closed_trades': 0
            }
        
        df = pd.DataFrame(self.trades)
        closed_df = df[df['status'] == 'CLOSED']
        
        if len(closed_df) == 0:
            return {
                'total_trades': len(df),
                'open_trades': len(df[df['status'] == 'OPEN']),
                'closed_trades': 0
            }
        
        return {
            'total_trades': len(df),
            'open_trades': len(df[df['status'] == 'OPEN']),
            'closed_trades': len(closed_df),
            'win_rate': (closed_df['pnl'] > 0).mean(),
            'total_pnl': closed_df['pnl'].sum(),
            'avg_return': closed_df['return_pct'].mean(),
            'max_return': closed_df['return_pct'].max(),
            'min_return': closed_df['return_pct'].min(),
            'avg_holding_days': closed_df['holding_days'].mean()
        }
    
    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        匯出交易記錄為 CSV
        
        Args:
            filename: 檔案名稱 (若為 None 則自動生成)
        
        Returns:
            CSV 檔案路徑
        """
        if filename is None:
            filename = f"trades_{self.current_session}.csv"
        
        filepath = self.log_dir / filename
        
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"交易記錄已匯出: {filepath}")
        else:
            logger.warning("無交易記錄可匯出")
        
        return filepath


# 測試程式碼
if __name__ == "__main__":
    # 初始化交易日誌
    tl = TradeLogger()
    
    # 測試記錄預測
    tl.log_prediction(
        date=datetime.now(),
        prediction={'action': 'BUY_CALL', 'confidence': 0.85},
        market_data={'close': 18000, 'atr': 200}
    )
    
    # 測試記錄進場
    trade_id = tl.log_trade_entry(
        date=datetime.now(),
        action='BUY_CALL',
        strike=18000,
        premium=100,
        confidence=0.85,
        position_size=5000,
        reason='模型預測上漲'
    )
    
    # 測試記錄出場
    tl.log_trade_exit(
        trade_id=trade_id,
        exit_date=datetime.now(),
        exit_premium=150,
        pnl=2500,
        return_pct=0.50,
        reason='達到停利點'
    )
    
    # 測試摘要
    summary = tl.get_session_summary()
    print("\n=== 交易摘要 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 匯出 CSV
    tl.export_to_csv()
