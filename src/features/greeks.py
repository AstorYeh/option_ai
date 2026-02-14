"""
Greeks 計算模組
使用 Black-Scholes 模型計算選擇權 Greeks
"""
import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlackScholesGreeks:
    """Black-Scholes Greeks 計算器"""
    
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.01,
        volatility: float = 0.2
    ):
        """
        初始化 Greeks 計算器
        
        Args:
            spot_price: 現貨價格
            strike_price: 履約價
            time_to_expiry: 到期時間(年)
            risk_free_rate: 無風險利率(年化)
            volatility: 波動率(年化)
        """
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        
        # 計算 d1 和 d2
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()
    
    def _calculate_d1(self) -> float:
        """計算 d1"""
        if self.T <= 0:
            return 0.0
        
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        return d1
    
    def _calculate_d2(self) -> float:
        """計算 d2"""
        return self.d1 - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        """計算 Call 理論價格"""
        if self.T <= 0:
            return max(0, self.S - self.K)
        
        price = self.S * norm.cdf(self.d1) - \
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return price
    
    def put_price(self) -> float:
        """計算 Put 理論價格"""
        if self.T <= 0:
            return max(0, self.K - self.S)
        
        price = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - \
                self.S * norm.cdf(-self.d1)
        return price
    
    def delta(self, option_type: str = 'call') -> float:
        """
        計算 Delta (價格敏感度)
        
        Delta 表示現貨價格變動 1 元時,選擇權價格的變動
        Call Delta: 0 ~ 1
        Put Delta: -1 ~ 0
        
        Args:
            option_type: 'call' 或 'put'
        
        Returns:
            Delta 值
        """
        if self.T <= 0:
            if option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        if option_type == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1
    
    def gamma(self) -> float:
        """
        計算 Gamma (Delta 變化率)
        
        Gamma 表示現貨價格變動 1 元時,Delta 的變動
        Gamma 對 Call 和 Put 相同
        
        Returns:
            Gamma 值
        """
        if self.T <= 0:
            return 0.0
        
        gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def theta(self, option_type: str = 'call') -> float:
        """
        計算 Theta (時間價值衰減)
        
        Theta 表示時間經過 1 天時,選擇權價格的變動
        通常為負值(買方不利)
        
        Args:
            option_type: 'call' 或 'put'
        
        Returns:
            Theta 值(日 Theta,已除以 365)
        """
        if self.T <= 0:
            return 0.0
        
        if option_type == 'call':
            theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) -
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        else:
            theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) +
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
        
        # 轉換為日 Theta
        return theta / 365
    
    def vega(self) -> float:
        """
        計算 Vega (波動率敏感度)
        
        Vega 表示波動率變動 1% 時,選擇權價格的變動
        Vega 對 Call 和 Put 相同
        
        Returns:
            Vega 值
        """
        if self.T <= 0:
            return 0.0
        
        vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        
        # 轉換為波動率變動 1% 的影響
        return vega / 100
    
    def rho(self, option_type: str = 'call') -> float:
        """
        計算 Rho (利率敏感度)
        
        Rho 表示利率變動 1% 時,選擇權價格的變動
        
        Args:
            option_type: 'call' 或 'put'
        
        Returns:
            Rho 值
        """
        if self.T <= 0:
            return 0.0
        
        if option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        
        # 轉換為利率變動 1% 的影響
        return rho / 100
    
    def get_all_greeks(self, option_type: str = 'call') -> dict:
        """
        一次性計算所有 Greeks
        
        Args:
            option_type: 'call' 或 'put'
        
        Returns:
            包含所有 Greeks 的字典
        """
        greeks = {
            'price': self.call_price() if option_type == 'call' else self.put_price(),
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'rho': self.rho(option_type)
        }
        
        return greeks


def calculate_implied_volatility(
    option_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    option_type: str = 'call',
    risk_free_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-5
) -> Optional[float]:
    """
    使用牛頓法計算隱含波動率
    
    Args:
        option_price: 市場選擇權價格
        spot_price: 現貨價格
        strike_price: 履約價
        time_to_expiry: 到期時間(年)
        option_type: 'call' 或 'put'
        risk_free_rate: 無風險利率
        max_iterations: 最大迭代次數
        tolerance: 容忍誤差
    
    Returns:
        隱含波動率,若無法收斂則返回 None
    """
    if time_to_expiry <= 0 or option_price <= 0:
        return None
    
    # 初始猜測值
    sigma = 0.2
    
    for i in range(max_iterations):
        # 計算理論價格
        bs = BlackScholesGreeks(spot_price, strike_price, time_to_expiry, risk_free_rate, sigma)
        
        if option_type == 'call':
            theoretical_price = bs.call_price()
        else:
            theoretical_price = bs.put_price()
        
        # 計算價格差異
        price_diff = theoretical_price - option_price
        
        # 檢查是否收斂
        if abs(price_diff) < tolerance:
            return sigma
        
        # 計算 Vega
        vega = bs.vega() * 100  # 轉回原始 Vega
        
        if vega == 0:
            return None
        
        # 牛頓法更新
        sigma = sigma - price_diff / vega
        
        # 確保波動率為正
        if sigma <= 0:
            sigma = 0.01
    
    logger.warning(f"隱含波動率計算未收斂: {option_price}, {spot_price}, {strike_price}")
    return None


def analyze_greeks_for_strategy(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str = 'call'
) -> dict:
    """
    分析 Greeks 對買方策略的影響
    
    Args:
        spot_price: 現貨價格
        strike_price: 履約價
        time_to_expiry: 到期時間(年)
        volatility: 波動率
        option_type: 'call' 或 'put'
    
    Returns:
        策略分析結果
    """
    bs = BlackScholesGreeks(spot_price, strike_price, time_to_expiry, 0.01, volatility)
    greeks = bs.get_all_greeks(option_type)
    
    analysis = {
        'greeks': greeks,
        'moneyness': 'ATM' if abs(spot_price - strike_price) < 100 else 
                     ('ITM' if (spot_price > strike_price and option_type == 'call') or 
                               (spot_price < strike_price and option_type == 'put') else 'OTM'),
        'time_decay_per_day': greeks['theta'],
        'breakeven_move': abs(greeks['price'] / greeks['delta']) if greeks['delta'] != 0 else None,
        'vega_exposure': greeks['vega'],
    }
    
    # 買方策略評估
    if time_to_expiry < 7/365:  # 少於 7 天
        analysis['time_risk'] = 'high'
        analysis['recommendation'] = '時間價值衰減快,不建議買方'
    elif time_to_expiry < 14/365:  # 7-14 天
        analysis['time_risk'] = 'medium'
        analysis['recommendation'] = '需要明確方向性,謹慎操作'
    else:
        analysis['time_risk'] = 'low'
        analysis['recommendation'] = '時間充裕,適合買方策略'
    
    # Vega 評估
    if greeks['vega'] > 50:
        analysis['volatility_sensitivity'] = 'high'
    elif greeks['vega'] > 20:
        analysis['volatility_sensitivity'] = 'medium'
    else:
        analysis['volatility_sensitivity'] = 'low'
    
    return analysis


# 測試程式碼
if __name__ == "__main__":
    # 測試 Greeks 計算
    spot = 18000
    strike = 18000
    time_to_expiry = 30 / 365  # 30 天
    volatility = 0.2
    
    print("=== Call Option Greeks ===")
    bs_call = BlackScholesGreeks(spot, strike, time_to_expiry, 0.01, volatility)
    call_greeks = bs_call.get_all_greeks('call')
    
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    print("\n=== Put Option Greeks ===")
    put_greeks = bs_call.get_all_greeks('put')
    
    for greek, value in put_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    # 測試隱含波動率計算
    print("\n=== Implied Volatility ===")
    market_price = call_greeks['price']
    iv = calculate_implied_volatility(market_price, spot, strike, time_to_expiry, 'call')
    print(f"Market Price: {market_price:.2f}")
    print(f"Implied Volatility: {iv:.2%}")
    
    # 策略分析
    print("\n=== Strategy Analysis ===")
    analysis = analyze_greeks_for_strategy(spot, strike, time_to_expiry, volatility, 'call')
    print(f"Moneyness: {analysis['moneyness']}")
    print(f"Time Risk: {analysis['time_risk']}")
    print(f"Recommendation: {analysis['recommendation']}")
