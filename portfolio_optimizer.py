

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.15):  # Risk-free rate parametre olarak alınabilir
        self.risk_free_rate = risk_free_rate

    def get_stock_data(self, symbols, period='1y'):
        """Hisse senedi verilerini çeker. Hata yönetimi eklenmiştir."""
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                stock = yf.Ticker(f"{symbol}.IS")
                hist = stock.history(period=period)
                data[symbol] = hist['Close']
            except Exception as e:
                print(f"Hata: {symbol} verisi çekilemedi: {e}")
                return None  # Hata durumunda None döndür
        return data

    def calculate_returns(self, data):
        """Günlük getirileri hesaplar. Veri kontrolü eklenmiştir."""
        if data.empty or len(data) < 2:
            print("Hata: Yetersiz veri.")
            return None
        return data.pct_change().dropna()  # İlk satırı (NaN) at

    def calculate_portfolio_metrics(self, weights, returns):
        """Portföy metriklerini hesaplar."""
        annual_factor = 252  # İş günü sayısı parametre olarak alınabilir
        portfolio_return = np.sum(returns.mean() * weights) * annual_factor
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * annual_factor, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk else -np.inf # 0 bölme hatası önlendi.

        return portfolio_return, portfolio_risk, sharpe_ratio

    def optimize_portfolio(self, returns, risk_tolerance):
        """Portföy optimizasyonu yapar. Kısıtlar ve algoritma iyileştirildi."""
        num_assets = returns.shape[1]
        initial_weights = np.array([1/num_assets] * num_assets)

        # Daha güvenli kısıtlar
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(num_assets)]  # Ağırlıklar 0-1 arasında

        def objective(weights):
            portfolio_return, portfolio_risk, _ = self.calculate_portfolio_metrics(weights, returns)
            return -(portfolio_return - risk_tolerance * portfolio_risk)  # Amaç fonksiyonu korunuyor.

        result = minimize(
            objective,
            initial_weights,
            method='trust-constr',  # veya 'SLSQP'
            constraints=constraints,
            bounds=bounds
        )

        return result.x

    def optimize(self, symbols, capital, risk_tolerance):
        """Ana optimizasyon fonksiyonu. Hata yönetimi eklendi."""
        data = self.get_stock_data(symbols)
        if data is None:
            return None

        returns = self.calculate_returns(data)
        if returns is None:
            return None

        optimal_weights = self.optimize_portfolio(returns, risk_tolerance)

        portfolio = {}
        for symbol, weight in zip(symbols, optimal_weights):
            if not np.isnan(weight):  # NaN kontrolü
                try:
                    estimated_shares = int((weight * capital) / data[symbol].iloc[-1]) if data[symbol].iloc[-1] != 0 else 0 # 0 bölme hatası kontrolü
                except (IndexError, KeyError):
                    estimated_shares = 0
                portfolio[symbol] = {
                    'weight': weight,
                    'amount': weight * capital,
                    'estimated_shares': estimated_shares
                }
            else:
                print(f"Uyarı: {symbol} için geçersiz ağırlık değeri (NaN).")

        return portfolio