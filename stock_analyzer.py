# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import yfinance as yf
# import requests
# from bs4 import BeautifulSoup
# import json

# class EnhancedBISTAnalyzer:
#     def __init__(self):
#         self.risk_limit = 0.02
#         self.portfolio_size = 5
#         self.fundamental_weights = {
#             'pe_ratio': 0.3,
#             'pb_ratio': 0.2,
#             'profit_margin': 0.25,
#             'debt_ratio': 0.25
#         }
        
#     def get_fundamental_data(self, symbol):
#         """Temel analiz verilerini çeker"""
#         try:
#             stock = yf.Ticker(f"{symbol}.IS")
#             info = stock.info
            
#             return {
#                 'pe_ratio': info.get('forwardPE', 0),
#                 'pb_ratio': info.get('priceToBook', 0),
#                 'profit_margin': info.get('profitMargins', 0),
#                 'debt_ratio': info.get('debtToEquity', 0),
#                 'dividend_yield': info.get('dividendYield', 0),
#                 'market_cap': info.get('marketCap', 0)
#             }
#         except:
#             return None

#     def calculate_technical_indicators(self, df):
#         """Genişletilmiş teknik indikatörler"""
#         # Mevcut indikatörler
#         df = super().calculate_indicators(df)
        
#         # MACD
#         exp1 = df['Close'].ewm(span=12, adjust=False).mean()
#         exp2 = df['Close'].ewm(span=26, adjust=False).mean()
#         df['MACD'] = exp1 - exp2
#         df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
#         # Stochastic Oscillator
#         low_min = df['Low'].rolling(window=14).min()
#         high_max = df['High'].rolling(window=14).max()
#         df['Stoch_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
#         df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
#         # Average True Range (ATR)
#         high_low = df['High'] - df['Low']
#         high_close = np.abs(df['High'] - df['Close'].shift())
#         low_close = np.abs(df['Low'] - df['Close'].shift())
#         ranges = pd.concat([high_low, high_close, low_close], axis=1)
#         true_range = np.max(ranges, axis=1)
#         df['ATR'] = true_range.rolling(window=14).mean()
        
#         return df

#     def calculate_risk_score(self, fundamental_data, technical_data):
#         """Risk skoru hesaplama"""
#         risk_score = 0
        
#         # Temel analiz risk faktörleri
#         if fundamental_data:
#             if fundamental_data['pe_ratio'] > 30: risk_score += 2
#             if fundamental_data['debt_ratio'] > 1: risk_score += 2
#             if fundamental_data['profit_margin'] < 0: risk_score += 1
            
#         # Teknik analiz risk faktörleri
#         volatility = technical_data['ATR'][-1] / technical_data['Close'][-1]
#         if volatility > 0.03: risk_score += 2
#         if technical_data['RSI'][-1] > 80 or technical_data['RSI'][-1] < 20: risk_score += 1
        
#         return min(risk_score, 10)  # 0-10 arası risk skoru

#     def generate_trade_signals(self, df, fundamental_data):
#         """Geliştirilmiş alım-satım sinyalleri"""
#         signals = pd.DataFrame(index=df.index)
#         signals['Signal'] = 0
        
#         # Teknik sinyaller
#         tech_signals = super().generate_signals(df)
        
#         # MACD sinyalleri
#         signals.loc[df['MACD'] > df['Signal_Line'], 'MACD_Signal'] = 1
#         signals.loc[df['MACD'] <= df['Signal_Line'], 'MACD_Signal'] = -1
        
#         # Stochastic sinyaller
#         signals.loc[(df['Stoch_K'] < 20) & (df['Stoch_D'] < 20), 'Stoch_Signal'] = 1
#         signals.loc[(df['Stoch_K'] > 80) & (df['Stoch_D'] > 80), 'Stoch_Signal'] = -1
        
#         # Temel analiz filtreleri
#         if fundamental_data:
#             good_fundamentals = (
#                 fundamental_data['pe_ratio'] < 20 and
#                 fundamental_data['debt_ratio'] < 1 and
#                 fundamental_data['profit_margin'] > 0
#             )
#         else:
#             good_fundamentals = True
        
#         # Sinyal birleştirme
#         signals['Signal'] = np.where(
#             (tech_signals['Signal'] == 1) & 
#             (signals['MACD_Signal'] == 1) & 
#             (signals['Stoch_Signal'] == 1) &
#             good_fundamentals,
#             1,
#             np.where(
#                 (tech_signals['Signal'] == -1) & 
#                 (signals['MACD_Signal'] == -1),
#                 -1,
#                 0
#             )
#         )
        
#         return signals

#     def generate_report(self, symbol, analysis_result):
#         """Detaylı analiz raporu oluşturur"""
#         report = {
#             'symbol': symbol,
#             'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
#             'technical_analysis': {
#                 'current_price': analysis_result['current_price'],
#                 'ma20': analysis_result['ma20'],
#                 'ma50': analysis_result['ma50'],
#                 'rsi': analysis_result['rsi'],
#                 'macd': analysis_result['macd'],
#                 'stochastic': {
#                     'k': analysis_result['stoch_k'],
#                     'd': analysis_result['stoch_d']
#                 }
#             },
#             'fundamental_analysis': analysis_result['fundamental_data'],
#             'risk_analysis': {
#                 'risk_score': analysis_result['risk_score'],
#                 'position_size': analysis_result['position_size'],
#                 'stop_loss': analysis_result['stop_loss']
#             },
#             'trade_recommendation': {
#                 'signal': analysis_result['signal'],
#                 'confidence': analysis_result['confidence'],
#                 'target_price': analysis_result['target_price']
#             }
#         }
        
#         return report

#     def save_to_file(self, report, filename):
#         """Raporu JSON formatında kaydeder"""
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(report, f, ensure_ascii=False, indent=4)

#     def analyze_stock(self, symbol, capital):
#         """Geliştirilmiş tam analiz"""
#         df = self.get_stock_data(symbol)
#         if df is None:
#             return None
            
#         fundamental_data = self.get_fundamental_data(symbol)
#         signals = self.generate_trade_signals(df, fundamental_data)
#         risk_score = self.calculate_risk_score(fundamental_data, df)
        
#         # Stop-loss ve hedef fiyat hesaplama
#         atr = df['ATR'][-1]
#         current_price = df['Close'][-1]
#         stop_loss = current_price - (2 * atr)
#         target_price = current_price + (3 * atr)  # Risk:Ödül oranı 1:1.5
        
#         # Güven skoru hesaplama
#         confidence = self.calculate_confidence_score(df, fundamental_data, risk_score)
        
#         analysis = {
#             'symbol': symbol,
#             'current_price': current_price,
#             'signal': signals['Signal'][-1],
#             'position_size': self.calculate_position_size(capital, current_price),
#             'risk_score': risk_score,
#             'fundamental_data': fundamental_data,
#             'rsi': df['RSI'][-1],
#             'ma20': df['MA20'][-1],
#             'ma50': df['MA50'][-1],
#             'macd': df['MACD'][-1],
#             'stoch_k': df['Stoch_K'][-1],
#             'stoch_d': df['Stoch_D'][-1],
#             'confidence': confidence,
#             'stop_loss': stop_loss,
#             'target_price': target_price
#         }
        
#         return analysis

# def main():
#     analyzer = EnhancedBISTAnalyzer()
    
#     # Örnek hisse listesi
#     stocks = ['THYAO', 'GARAN', 'ASELS', 'KCHOL', 'EREGL']
#     capital = 100000
    
#     for symbol in stocks:
#         analysis = analyzer.analyze_stock(symbol, capital)
#         if analysis:
#             report = analyzer.generate_report(symbol, analysis)
#             analyzer.save_to_file(report, f'{symbol}_analysis.json')
#             print(f"{symbol} analizi tamamlandı.")

# if __name__ == "__main__":
#     main()






import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import json

class EnhancedBISTAnalyzer:
    def __init__(self):
        self.risk_limit = 0.02  # Risk limiti (sermayenin %2'si)
        self.portfolio_size = 5  # Portföydeki hisse senedi sayısı
        self.fundamental_weights = {  # Temel analiz faktörlerinin ağırlıkları
            'pe_ratio': 0.3,
            'pb_ratio': 0.2,
            'profit_margin': 0.25,
            'debt_ratio': 0.25
        }

    def get_stock_data(self, symbol, period='1y'): # EKLENDİ
        """Hisse senedi verilerini çeker."""
        try:
            stock = yf.Ticker(f"{symbol}.IS")
            hist = stock.history(period=period)
            return hist  # DataFrame döndürüyor
        except Exception as e:
            print(f"Hata: {symbol} hisse verisi çekilemedi: {e}")
            return None

    def get_fundamental_data(self, symbol):
        """Temel analiz verilerini çeker. Hata yönetimi eklenmiştir."""
        try:
            stock = yf.Ticker(f"{symbol}.IS")
            info = stock.info
            return {
                'pe_ratio': info.get('forwardPE'),  # Varsayılan değer olarak None kullanmak daha iyi olabilir.
                'pb_ratio': info.get('priceToBook'),
                'profit_margin': info.get('profitMargins'),
                'debt_ratio': info.get('debtToEquity'),
                'dividend_yield': info.get('dividendYield'),
                'market_cap': info.get('marketCap')
            }
        except Exception as e:
            print(f"Hata: {symbol} temel verisi çekilemedi: {e}")
            return None

    def calculate_indicators(self, df):
        """Temel indikatörleri hesaplar."""
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], window=14)  # Örnek RSI hesaplama
        return df

    def calculate_rsi(self, series, window=14):
        """RSI hesaplar."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window - 1, adjust=False).mean()
        ema_down = down.ewm(com=window - 1, adjust=False).mean()
        rsi = np.where(ema_down == 0, 100, 100 - (100 * ema_down / ema_up))
        return rsi

    def calculate_technical_indicators(self, df):
        """Genişletilmiş teknik indikatörler. Veri kontrolü eklenmiştir."""
        if df.empty or len(df) < 20:  # Veri kontrolü eklendi
            print("Hata: Teknik indikatörler için yetersiz veri.")
            return df

        df = self.calculate_indicators(df)  # Düzeltilmiş süper kullanımı

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        return df

    def calculate_risk_score(self, fundamental_data, technical_data):
        """Risk skoru hesaplama."""
        risk_score = 0

        # Temel analiz risk faktörleri
        if fundamental_data:
            pe_ratio = fundamental_data.get('pe_ratio')  # None olabilir
            debt_ratio = fundamental_data.get('debt_ratio')  # None olabilir
            profit_margin = fundamental_data.get('profit_margin')  # None olabilir

            if pe_ratio is not None and pe_ratio > 30: risk_score += 2
            if debt_ratio is not None and debt_ratio > 1: risk_score += 2
            if profit_margin is not None and profit_margin < 0: risk_score += 1

        # Teknik analiz risk faktörleri
        if not technical_data.empty:  # Teknik veri kontrolü
            volatility = technical_data['ATR'].iloc[-1] / technical_data['Close'].iloc[-1] # .iloc eklendi
            if volatility > 0.03: risk_score += 2
            if technical_data['RSI'].iloc[-1] > 80 or technical_data['RSI'].iloc[-1] < 20: risk_score += 1 # .iloc eklendi

        return min(risk_score, 10)  # 0-10 arası risk skoru

    def calculate_confidence_score(self, df, fundamental_data, risk_score):
        """Güven skoru hesaplama (örnek)."""
        confidence = 100 - risk_score * 5  # Örnek bir güven skoru hesaplama
        return confidence

    def calculate_position_size(self, capital, current_price):
        """Pozisyon büyüklüğü hesaplama."""
        position_size = capital * self.risk_limit / current_price  # Risk limitine göre pozisyon büyüklüğü
        return position_size

    def generate_signals(self, df):
        """Alım satım sinyalleri üretme (örnek)."""
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0
        signals.loc[df['MA20'] > df['MA50'], 'Signal'] = 1
        signals.loc[df['MA20'] <= df['MA50'], 'Signal'] = -1
        return signals

    def generate_trade_signals(self, df, fundamental_data):
        """Geliştirilmiş alım-satım sinyalleri."""
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0

        # Teknik sinyaller
        tech_signals = self.generate_signals(df)  # Düzeltilmiş süper kullanımı

        # MACD sinyalleri
        signals.loc[df['MACD'] > df['Signal_Line'], 'MACD_Signal'] = 1
        signals.loc[df['MACD'] <= df['Signal_Line'], 'MACD_Signal'] = -1

        # Stochastic sinyaller
        signals.loc[(df['Stoch_K'] < 20) & (df['Stoch_D'] < 20), 'Stoch_Signal'] = 1
        signals.loc[(df['Stoch_K'] > 80) & (df['Stoch_D'] > 80), 'Stoch_Signal'] = -1

        # Temel analiz filtreleri
        if fundamental_data:
            debt_ratio = fundamental_data.get('debt_ratio')  # Varsayılan değer None
            good_fundamentals = (
                fundamental_data.get('pe_ratio', np.inf) < 20 and
                (debt_ratio is not None and debt_ratio < 1) and # None kontrolü
                fundamental_data.get('profit_margin', -np.inf) > 0
            )
        else:
            good_fundamentals = True

        # Sinyal birleştirme (örnek mantık)
        signals['Signal'] = np.where(
            (tech_signals['Signal'] == 1) &
            (signals['MACD_Signal'] == 1) &
            (signals['Stoch_Signal'] == 1) &
            good_fundamentals,
            1,
            np.where(
                (tech_signals['Signal'] == -1) &
                (signals['MACD_Signal'] == -1),
                -1,
                0
            )
        )

        return signals

    def generate_report(self, symbol, analysis_result):
        """Detaylı analiz raporu oluşturur. NumPy/Pandas tipleri dönüştürülüyor."""
        report = {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'technical_analysis': {
                'current_price': analysis_result.get('current_price'),
                'ma20': analysis_result.get('ma20'),
                'ma50': analysis_result.get('ma50'),
                'rsi': analysis_result.get('rsi'),
                'macd': analysis_result.get('macd'),
                'stochastic': {
                    'k': analysis_result.get('stoch_k'),
                    'd': analysis_result.get('stoch_d')
                }
            },
            'fundamental_analysis': analysis_result.get('fundamental_data'),
            'risk_analysis': {
                'risk_score': analysis_result.get('risk_score'),
                'position_size': analysis_result.get('position_size'),
                'stop_loss': analysis_result.get('stop_loss')
            },
            'trade_recommendation': {
                'signal': analysis_result.get('signal'),
                'confidence': analysis_result.get('confidence'),
                'target_price': analysis_result.get('target_price')
            }
        }

        # NumPy/Pandas tiplerini Python tiplerine dönüştür
        report = self._convert_numpy_pandas_to_python(report)  # Yardımcı fonksiyon eklendi

        return report
    

    def _convert_numpy_pandas_to_python(self, obj):  # Yardımcı fonksiyon
        """NumPy/Pandas objelerini Python objelerine dönüştürür."""
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()  # NumPy sayılarını Python'a dönüştür
        elif isinstance(obj, pd.Series):
            return obj.to_list()  # Pandas Series'i listeye dönüştür
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')  # Pandas DataFrame'i liste
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_pandas_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_pandas_to_python(elem) for elem in obj]
        else:
            return obj

    def save_to_file(self, report, filename):
        """Raporu JSON formatında kaydeder."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

    def analyze_stock(self, symbol, capital):  # analyze_stock sınıfın içine taşındı
        """Geliştirilmiş tam analiz."""

        fundamental_data = self.get_fundamental_data(symbol)
        if fundamental_data is None:
            return None

        df = self.get_stock_data(symbol)
        if df is None or df.empty:
            return None

        df = self.calculate_technical_indicators(df)
        if df.empty:
            return None

        signals = self.generate_trade_signals(df, fundamental_data)
        risk_score = self.calculate_risk_score(fundamental_data, df)

        # Stop-loss ve hedef fiyat hesaplama
        atr = df['ATR'].iloc[-1] if not df['ATR'].empty else 0
        current_price = df['Close'].iloc[-1] if not df['Close'].empty else 0
        stop_loss = current_price - (2 * atr)
        target_price = current_price + (3 * atr)

        # Güven skoru hesaplama
        confidence = self.calculate_confidence_score(df, fundamental_data, risk_score)

        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': signals['Signal'].iloc[-1] if not signals.empty else 0,
            'position_size': self.calculate_position_size(capital, current_price),
            'risk_score': risk_score,
            'fundamental_data': fundamental_data,
            'rsi': df['RSI'].iloc[-1] if not df['RSI'].empty else 0,
            'ma20': df['MA20'].iloc[-1] if not df['MA20'].empty else 0,
            'ma50': df['MA50'].iloc[-1] if not df['MA50'].empty else 0,
            'macd': df['MACD'].iloc[-1] if not df['MACD'].empty else 0,
            'stoch_k': df['Stoch_K'].iloc[-1] if not df['Stoch_K'].empty else 0,
            'stoch_d': df['Stoch_D'].iloc[-1] if not df['Stoch_D'].empty else 0,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'target_price': target_price
        }

        return analysis

def main():
    analyzer = EnhancedBISTAnalyzer()

    stocks = ['THYAO', 'GARAN', 'ASELS', 'KCHOL', 'EREGL']
    capital = 100000

    for symbol in stocks:
        analysis = analyzer.analyze_stock(symbol, capital)  # analyzer nesnesi üzerinden çağrılıyor
        if analysis:
            report = analyzer.generate_report(symbol, analysis)
            analyzer.save_to_file(report, f'{symbol}_analysis.json')
            print(f"{symbol} analizi tamamlandı.")
        else:
            print(f"{symbol} analizi başarısız.")

if __name__ == "__main__":
    main()