import mysql.connector
from mysql.connector import Error
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json
import os

def insert_into_db(data):
    try:
        load_dotenv()
        # Pridobi nastavitve iz okolja
        db_host = os.getenv('DB_HOST')
        db_database = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')

        # Povezava z MySQL bazo
        conn = mysql.connector.connect(
            host=db_host,            # Nastavitev iz .env datoteke
            database=db_database,    # Nastavitev iz .env datoteke
            user=db_user,            # Nastavitev iz .env datoteke
            password=db_password     # Nastavitev iz .env datoteke
        )

        if conn.is_connected():
            cursor = conn.cursor()

            cursor.execute("DELETE FROM Companies")
            conn.commit()

            # Vstavljanje podatkov v bazo
            for entry in data:
                cursor.execute(''' 
                INSERT INTO Companies (uvoz, podjetje, prihodek, sprememba_prihodka, dobicek, sprememba_dobicka,
                sredstva, zaposleni, rang_drzave, leta_na_seznamu, povezava, TimeStamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CONVERT_TZ(NOW(), '+00:00', '+01:00'))
                ''', (
                    entry["uvoz"],
                    entry["podjetje"],
                    entry["prihodek"],
                    entry["sprememba_prihodka"],
                    entry["dobicek"],
                    entry["sprememba_dobicka"],
                    entry["sredstva"],
                    entry["zaposleni"],
                    entry["rang_drzave"],
                    entry["leta_na_seznamu"],
                    entry["povezava"]
                ))

            conn.commit()
            return None  # Ni napake, vračamo None
    except Error as e:
        return f"Napaka pri vstavljanju podatkov v MySQL: {e}"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def scrapeData(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, "Ni uspelo pridobiti strani"

    # Parsiranje strani
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []
    
    # Zajame prvih 20 vnosov
    for i, row in enumerate(soup.find_all("tr", {"data-cy": "list-row"})):
        if i >= 20:
            break
        cells = row.find_all("td")
        entry = {
            "uvoz": cells[0].text.strip(),
            "podjetje": cells[1].text.strip(),
            "prihodek": cells[2].text.strip(),
            "sprememba_prihodka": cells[3].text.strip(),
            "dobicek": cells[4].text.strip(),
            "sprememba_dobicka": cells[5].text.strip(),
            "sredstva": cells[6].text.strip(),
            "zaposleni": cells[7].text.strip(),
            "rang_drzave": cells[8].text.strip(),
            "leta_na_seznamu": cells[9].text.strip(),
            "povezava": cells[10].find("a")["href"] if cells[10].find("a") else None
        }
        data.append(entry)

    # Shrani podatke v JSON datoteko
    with open('fortune500.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    # Klic funkcije za vstavljanje v bazo
    error = insert_into_db(data)

    if error:
        return None, error  # Vrnemo napako, če je prišlo do nje
    else:
        return data, "Scrapanje uspešno, podatki naloženi v bazo"  # Če ni napake, vrnemo podatke in sporočilo o uspehu
