# risk_manager.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

class PortfolioRiskManager:
    def __init__(self):
        self.confidence_level = 0.95
        self.time_horizon = 10  # günlük VaR hesaplaması için
        
    def calculate_var(self, portfolio, returns):
        """Value at Risk (VaR) hesaplaması"""
        portfolio_returns = np.sum(returns * list(portfolio.values()), axis=1)
        var = norm.ppf(1 - self.confidence_level) * np.std(portfolio_returns) * \
              np.sqrt(self.time_horizon)
        return abs(var)
    
    def calculate_stress_test(self, portfolio, returns):
        """Stres testi senaryoları"""
        scenarios = {
            'market_crash': -0.20,  # %20 düşüş
            'moderate_decline': -0.10,  # %10 düşüş
            'best_case': 0.10,  # %10 yükseliş
        }
        
        results = {}
        portfolio_value = sum(portfolio.values())
        
        for scenario, change in scenarios.items():
            impact = portfolio_value * (1 + change)
            results[scenario] = impact
            
        return results
    
    def calculate_risk_metrics(self, portfolio, symbols):
        """Tüm risk metriklerini hesaplar"""
        data = self._get_historical_data(symbols)
        returns = data.pct_change().dropna()
        
        var = self.calculate_var(portfolio, returns)
        stress_test = self.calculate_stress_test(portfolio, returns)
        
        # Volatilite hesaplaması
        volatility = returns.std() * np.sqrt(252)  # Yıllık volatilite
        
        # Sharpe oranı hesaplaması
        risk_free_rate = 0.15  # Türkiye için yaklaşık
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility
        
        return {
            'var': var,
            'stress_test': stress_test,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _get_historical_data(self, symbols, period='1y'):
        """Geçmiş verileri çeker"""
        data = pd.DataFrame()
        
        for symbol in symbols:
            stock = yf.Ticker(f"{symbol}.IS")
            hist = stock.history(period=period)
            data[symbol] = hist['Close']
            
        return data
    
    def generate_risk_report(self, portfolio, symbols):
        """Detaylı risk raporu oluşturur"""
        metrics = self.calculate_risk_metrics(portfolio, symbols)
        
        report = {
            'summary': {
                'total_value': sum(portfolio.values()),
                'number_of_stocks': len(portfolio),
                'risk_level': self._determine_risk_level(metrics['volatility'])
            },
            'risk_metrics': {
                'value_at_risk': metrics['var'],
                'annual_volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio']
            },
            'stress_test': metrics['stress_test'],
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _determine_risk_level(self, volatility):
        """Risk seviyesini belirler"""
        if volatility < 0.15:
            return 'Düşük'
        elif volatility < 0.25:
            return 'Orta'
        else:
            return 'Yüksek'
    
    def _generate_recommendations(self, metrics):
        """Risk metriklerine göre öneriler oluşturur"""
        recommendations = []
        
        if metrics['var'] > 0.1:
            recommendations.append(
                "Portföy riski yüksek. Daha fazla çeşitlendirme önerilir."
            )
            
        if metrics['sharpe_ratio'] < 0.5:
            recommendations.append(
                "Risk-getiri oranı düşük. Daha az riskli varlıklara yönelim önerilir."
            )
            
        if metrics['volatility'] > 0.25:
            recommendations.append(
                "Portföy volatilitesi yüksek. Defansif hisselere ağırlık verilebilir."
            )
            
        return recommendations