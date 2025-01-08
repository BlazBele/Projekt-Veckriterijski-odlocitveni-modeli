# Večparametrsko odločitveno orodje

Aplikacija je izdelana z uporabo **Flaska** za spletno aplikacijo in uporablja **MongoDB** kot podatkovno bazo.

## Funkcionalnosti
- **AHP:** Analiza in vizualizacija hierarhičnih struktur za odločanje.
- **WSM:** Odločanje na osnovi uteži in točk.
- **Promethee:** Metoda za razvrščanje alternativ na osnovi preferenc.
- **Topsis:** Identifikacija najboljše alternative glede na idealno rešitev.
- **Scraping:** Vključuje orodja za pridobivanje podatkov, kot je `scrapper_fortune500.py`.

## Tehnologije
- **Backend:** Flask
- **Frontend:** HTML predloge (Jinja2) in Bootstrap za osnovno oblikovanje.
- **Frontend:** Html, Bootstrap za osnovno oblikovanje.
- **Podatkovna baza:** MongoDB za shranjevanje in pridobivanje podatkov.
- **Programski jezik:** Python

## Struktura datotek
- `app.py`: Glavna aplikacija Flask.
- `ahp.html`, `promethee.html`, `topsis.html`, `wsm.html`: HTML predloge za različne metode.
- `scraping.py`, `scrapper_fortune500.py`: Skripte za pridobivanje podatkov.
- `.env`: Konfiguracijska datoteka (npr. za povezavo z MongoDB).
- `fortune500.json`: Primer podatkov.

## Navodila za zagon
1. Namesti potrebne knjižnice iz `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
2. Ustvari datoteko `.env` in dodaj potrebne spremenljivke (npr. URL za MongoDB).
3. Zaženi aplikacijo:
    python app.py
4. Odpri aplikacijo v brskalniku na `http://localhost:5000`.
