import pymongo
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import os

# Funkcija za povezavo z MongoDB
def get_mongo_connection():
    load_dotenv()
    # Pridobi nastavitve iz okolja
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    return db

# Funkcija za vstavljanje podatkov v MongoDB
def insert_into_db(data):
    try:
        db = get_mongo_connection()
        companies_collection = db["Companies"]
        companies_collection.delete_many({}) 

        # Vstavi nove ali posodobi obstoječe podatke v zbirki
        for entry in data:
            companies_collection.update_one(
                {"povezava": entry["povezava"]},  # Poiščemo podjetje po povezavi
                {"$set": entry},  # Posodobimo podjetje s temi podatki
            )

        print("Podatki so bili uspešno vstavljeni v MongoDB.")
        return None  # Ni napake, vračamo None

    except Exception as e:
        return f"Napaka pri vstavljanju podatkov v MongoDB: {e}"

# Funkcija za zajemanje podatkov iz strani
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
