# Sistem za podporo odločanju za analizo investicij

To je spletna aplikacija za analizo investicij, ki omogoča uporabnikom primerjavo podjetij iz nabora Fortune500 s pomočjo metod večkriterijskega odločanja (MCDA). Aplikacija je enostavna za uporabo in ponuja orodja za vizualizacijo rezultatov ter prilagodljivo ocenjevanje.

---

## Lastnosti

- **Analiza investicij**: Podpora pri sprejemanju odločitev na podlagi ključnih kriterijev.
- **Večkriterijsko odločanje (MCDA)**: Metode AHP, TOPSIS, PROMETHEE in WSM.
- **Vizualizacija rezultatov**: Grafični prikazi za boljše razumevanje.
- **Uporabniku prijazna zasnova**: Intuitivni uporabniški vmesnik, ki omogoča enostavno navigacijo.
- **Prilagodljive analize**: Možnost izbire kriterijev in uteži za individualne potrebe uporabnikov.

---

## Struktura projekta

```plaintext
mvop/
│
├── baza.py                     # Skripta za upravljanje podatkovne baze
├── scrapper_fortune500.py      # Skripta za pridobivanje podatkov iz spletne strani
├── fortune500.json             # Podatkovna datoteka
│
├── projekt/                    # Glavna mapa projekta
│   ├── app.py                  # Glavna aplikacija
│   ├── scraping.py             # Orodja za obdelavo podatkov
│   ├── .env                    # Konfiguracijska datoteka okolja
│   ├── templates/              # HTML predloge
│   │   ├── base.html           # Osnovna predloga
│   │   ├── index.html          # Domača stran
│   │   ├── ahp.html            # Stran za analizo z metodo AHP
│   │   ├── topsis.html         # Stran za analizo z metodo TOPSIS
│   │   ├── promethee.html      # Stran za analizo z metodo PROMETHEE
│   │   ├── wsm.html            # Stran za analizo z metodo WSM
│   │   └── upload.html         # Stran za nalaganje podatkov
│   └── __pycache__/            # Predpomnilnik za Python
│
└── .git/                       # Nastavitve Git repozitorija
```

---

## Namestitev

### Zahteve

- Python 3.8 ali novejši
- Pip za upravljanje paketov
- Virtualno okolje (npr. `venv`)

### Koraki za namestitev

1. Klonirajte repozitorij:
   ```bash
   git clone https://github.com/BlazBele/Projekt-Veckriterijski-odlocitveni-procesov.git
   ```
2. Premaknite se v mapo projekta:
   ```bash
   cd mvop/projekt
   ```
3. Ustvarite in aktivirajte virtualno okolje:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Na Windows: venv\Scripts\activate
   ```
4. Namestite odvisnosti:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uporaba

1. Prilagodite konfiguracijo v `.env` datoteki.
2. Zaženite aplikacijo:
   ```bash
   python app.py
   ```
3. Dostopajte do aplikacije na naslovu `http://127.0.0.1:5000`.

---

## O aplikaciji

Aplikacija omogoča hitro in enostavno analizo podjetij na podlagi ključnih kriterijev:

- **Prihodek**: Ključni kazalnik uspešnosti podjetja.
- **Dobiček**: Merilo finančne stabilnosti.
- **Sredstva**: Celotno premoženje podjetja.
- **Število zaposlenih**: Indikator obsega in rasti podjetja.

Aplikacija vključuje metode večkriterijskega odločanja (MCDA), ki omogočajo oceno in rangiranje podjetij glede na zgoraj omenjene kriterije. Na voljo so naslednje metode:

- **AHP (Analitični hierarhični proces)**: Strukturna analiza na podlagi parnih primerjav.
- **TOPSIS**: Analiza na podlagi bližine idealni rešitvi.
- **PROMETHEE**: Primerjava na podlagi preferenčnih funkcij.
- **WSM (Model utežene vsote)**: Preprosta metoda na podlagi uteži.
