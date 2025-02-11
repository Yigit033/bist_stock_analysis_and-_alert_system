import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()  # .env dosyasını yükle

def create_prices_table(db_config):
    mydb = mysql.connector.connect(**db_config)
    mycursor = mydb.cursor()

    try:
        mycursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(255),
                price DECIMAL(10, 2),
                timestamp DATETIME
            )
        """)
        mydb.commit()
        print("prices tablosu oluşturuldu.")
    except Exception as e:
        print(f"Hata: {e}")

    mydb.close()

if __name__ == "__main__":
    db_config = {
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "host": os.getenv("MYSQL_HOST"),
    }
    create_prices_table(db_config)