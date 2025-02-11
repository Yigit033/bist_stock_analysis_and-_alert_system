# my_app.py


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import mysql.connector
from dotenv import load_dotenv
from stock_analyzer import EnhancedBISTAnalyzer
from portfolio_optimizer import PortfolioOptimizer
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import threading


# Ortam değişkenlerini yükle
load_dotenv()

# MySQL Bağlantı Ayarları
db_config = {
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "host": os.getenv("MYSQL_HOST"),
}

# Veritabanına Bağlan
mydb = mysql.connector.connect(**db_config)
mycursor = mydb.cursor()

st.subheader("Canlı Fiyatlar")
mycursor.execute("SELECT * FROM prices ORDER BY timestamp DESC")  # En son verileri çek
prices = mycursor.fetchall()
for price in prices:
    st.write(f"{price[0]}: {price[1]}")
mydb.close()

# E-posta ayarları (Güvenlik için ortam değişkenleri kullanılmalı)
# SENDER_EMAIL = os.getenv("SENDER_EMAIL")
# EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

analyzer = EnhancedBISTAnalyzer()
portfolio_optimizer = PortfolioOptimizer()

st.title("BIST Hisse Analiz ve Uyarı Sistemi")

# İzlenen Hisse Senetleri
watched_stocks = st.session_state.get("watched_stocks", {})
alert_settings = st.session_state.get("alert_settings", {})

# Hisse Ekleme/Çıkarma
col1, col2 = st.columns(2)
with col1:
    add_symbol = st.text_input("İzlenecek Hisse Kodu (Örn: THYAO.IS)")
    if st.button("Ekle"):
        if add_symbol:
            watched_stocks[add_symbol] = {}
            st.session_state.watched_stocks = watched_stocks
            st.success(f"{add_symbol} eklendi.")

with col2:
    remove_symbol = st.text_input("Çıkarılacak Hisse Kodu")
    if st.button("Çıkar"):
        if remove_symbol in watched_stocks:
            del watched_stocks[remove_symbol]
            st.session_state.watched_stocks = watched_stocks
            st.warning(f"{remove_symbol} çıkarıldı.")

# Uyarı Ayarları
st.subheader("Uyarı Ayarları")
alert_symbol = st.selectbox("Uyarı Ayarı Yapılacak Hisse", list(watched_stocks.keys()))
alert_type = st.selectbox("Uyarı Tipi", ["above", "below"])
alert_price = st.number_input("Uyarı Fiyatı")
alert_email = st.text_input("Uyarı E-posta Adresi")
if st.button("Uyarı Ekle"):
    if alert_symbol and alert_type and alert_price and alert_email:
        if alert_symbol not in alert_settings:
            alert_settings[alert_symbol] = []
        alert_settings[alert_symbol].append({
            "type": alert_type,
            "price": alert_price,
            "email": alert_email
        })
        st.session_state.alert_settings = alert_settings
        st.success("Uyarı eklendi.")

# Canlı Fiyatlar (MySQL)
st.subheader("Canlı Fiyatlar")
mydb = mysql.connector.connect(**db_config)
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM prices ORDER BY timestamp DESC")  # En son verileri çek
prices = mycursor.fetchall()
for price in prices:
    st.write(f"{price[0]}: {price[1]}")
mydb.close()

# Hisse Fiyatlarını Güncelle
if watched_stocks:
    for symbol in watched_stocks:
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.info.get('regularMarketPrice', 'Bilinmiyor')
            st.write(f"{symbol}: {current_price}")
        except Exception as e:
            st.error(f"Hata: {e}")

# Portföy Optimizasyonu
st.subheader("Portföy Optimizasyonu")
portfolio_symbols = st.multiselect("Portföye Eklenecek Hisse Senetleri", list(watched_stocks.keys()))
capital = st.number_input("Başlangıç Sermayesi", value=100000)
risk_tolerance = st.slider("Risk Toleransı (0-1)", 0.0, 1.0, 0.5)

if st.button("Portföyü Optimize Et"):
    if portfolio_symbols and capital > 0:
        try:
            optimal_weights = portfolio_optimizer.optimize(portfolio_symbols, capital, risk_tolerance)
            st.write("Optimal Portföy Ağırlıkları:")
            st.write(optimal_weights)
        except Exception as e:
            st.error(f"Optimizasyon Hatası: {e}")

# Hisse Analizi
st.subheader("Hisse Analizi")
analysis_symbol = st.text_input("Analiz Edilecek Hisse Kodu")
analysis_capital = st.number_input("Analiz için Sermaye", value=100000)

if st.button("Hisse Analizi Yap"):
    if analysis_symbol:
        try:
            analysis_results = analyzer.analyze_stock(analysis_symbol, analysis_capital)
            st.write("Hisse Analizi Sonuçları:")
            st.write(analysis_results)
        except Exception as e:
            st.error(f"Analiz Hatası: {e}")


def run_live_price_tracker():
    os.system("python live_price_tricker.py")

# Yeni thread ile çalıştır
thread = threading.Thread(target=run_live_price_tracker)
thread.start()