import requests
from bs4 import BeautifulSoup
import json

#URL Fortune Global 500 strani
url = "https://fortune.com/ranking/global500/search/?fortune500_y_n=true"

#Nastavi zaglavja, da simulira zahtevo brskalnika
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
}

#Pošlji zahtevo na stran
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("Ni uspelo pridobiti strani")
    exit()

#Parsiraj vsebino strani
soup = BeautifulSoup(response.text, 'html.parser')

#Izvleči vrstice podatkov z omejitvijo 20 vnosov
data = []
for i, row in enumerate(soup.find_all("tr", {"data-cy": "list-row"})):
    if i >= 20:  # Ustavi po 20 vnosih
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

#Zapiši podatke v JSON datoteko
with open('fortune500.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print("Prvih 20 vnosov uspešno shranjenih v fortune500.json")
