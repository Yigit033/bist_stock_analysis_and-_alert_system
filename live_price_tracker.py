import yfinance as yf
import pandas as pd
import time
import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def get_and_save_prices(symbols, db_config):
    try:  # Veritabanı bağlantısı için try bloğu
        mydb = mysql.connector.connect(**db_config)
        mycursor = mydb.cursor()

        for symbol in symbols:
            try:  # Her hisse senedi için ayrı try bloğu
                stock = yf.Ticker(symbol)
                current_price = stock.info.get('regularMarketPrice') # .get() ile None kontrolü

                if current_price is None:  # Fiyat yoksa veritabanına kaydetme
                    print(f"{symbol} için fiyat bilgisi bulunamadı.")
                    continue  # Sonraki hisse senedine geç

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                sql = "INSERT INTO prices (symbol, price, timestamp) VALUES (%s, %s, %s)"
                val = (symbol, current_price, timestamp)
                mycursor.execute(sql, val)
                mydb.commit()
                print(f"{symbol} fiyatı kaydedildi: {current_price}") # Başarı mesajı

            except Exception as e:  # Hisse senedi verisi çekme hatası
                print(f"{symbol} için hata: {e}")
                continue # Sonraki hisse senedine geç

        mydb.close()

    except mysql.connector.Error as err: # Veritabanı bağlantı hatası
        if err.errno == 2003: # Bağlantı hatası ise
            print("Veritabanı bağlantı hatası. Lütfen veritabanı bilgilerini kontrol edin ve sunucunun çalıştığından emin olun.")
        else:
            print(f"Veritabanı hatası: {err}")

if __name__ == "__main__":
    symbols = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "KCHOL.IS", "EREGL.IS"]  # İzlenen hisse senetleri
    db_config = {
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "host": os.getenv("MYSQL_HOST")

    }
    while True:
        get_and_save_prices(symbols, db_config)
        time.sleep(60)  # 60 saniyede bir güncelle (isteğe göre ayarlanabilir)
        print("Veriler güncellendi.") # Döngü çalıştığında mesaj