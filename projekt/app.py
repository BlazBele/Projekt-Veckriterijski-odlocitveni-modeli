from flask import Flask, render_template, request, jsonify, session
from scraping import scrapeData
import mysql.connector
import json
import os
import numpy as np
from dotenv import load_dotenv

# Nalaganje okolijskih spremenljivk iz .env datoteke
load_dotenv()

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Konfiguracija za povezavo z MySQL bazo
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'port': os.getenv('DB_PORT')
}

@app.route("/")
def index():
    """Prikaže glavno stran s tabelo podjetij iz baze."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM Companies")
    companies = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template("index.html", companies=companies)



@app.route("/scrape", methods=["POST"])
def scrape():
    """API endpoint za scrapanje podatkov in shranjevanje v JSON."""
    url = request.json.get("url")
    if not url:
        return jsonify({"status": "error", "message": "Manjka URL"}), 400

    data, message = scrapeData(url)
    if data is None:
        return jsonify({"status": "error", "message": message}), 500

    return jsonify({"status": "success", "message": message, "data": data}), 200


@app.route("/topsis")
def topsis():
    """Stran za metodo TOPSIS."""
    return render_template("topsis.html")

@app.route("/promethee")
def promethee():
    """Stran za metodo PROMETHEE."""
    return render_template("promethee.html")

@app.route("/wsm")
def wsm():
    """Stran za metodo WSM."""
    return render_template("wsm.html")

@app.route("/upload")
def upload():
    """Stran za metodo Upload"""
    return render_template("upload.html")


# Nalaganje podatkov iz JSON datoteke
def load_data():
    with open('fortune500.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/ahp', methods=['GET', 'POST'])
def ahp():
    companies = load_data()
    
    # Kriteriji, ki so na voljo (moramo zagotoviti, da so imena v teh kriterijih v skladu s podatki)
    criteria = [
        'Prihodek',
        'Sprememba prihodka',
        'Dobiček',
        'Sprememba dobička',
        'Sredstva',
        'Zaposleni',
        'Rang države',
        'Leta na seznamu'
    ]
    
    # 1. Korak: Izbira kriterijev
    if request.method == 'GET' or request.form.get('data') == 'selection':
        return render_template('ahp.html', data='selection', criteria=criteria)
    
    # 2. Korak: Izbira podjetij
    if request.form.get('data') == 'companies':
        selected_criteria = request.form.getlist('selected_criteria')
        session['selected_criteria'] = selected_criteria
        
        # Tukaj bomo prilagodili filtriranje podjetij
        # Predpostavimo, da so imena v `fortune500.json` v manjših črkah (npr. 'prihodek', 'dobicek')
        selected_companies = []
        for company in companies:
            if any(company.get(criterion.lower()) for criterion in selected_criteria):
                selected_companies.append(company)
        
        session['selected_companies'] = selected_companies

        return render_template('ahp.html', data='companies', companies=selected_companies)

    # 3. Korak: Vnos parnih primerjav (uteži)
    if request.form.get('data') == 'weights':
        selected_criteria = session.get('selected_criteria', [])
        num_criteria = len(selected_criteria)

        # Preberi podatke iz tabele parnih primerjav
        pairwise_data = []
        for i in range(num_criteria):
            row = []
            for j in range(num_criteria):
                if i == j:
                    row.append(1.0)  # Diagonalne vrednosti so vedno 1
                elif i < j:
                    try:
                        val = float(request.form.get(f"pairwise_{i}_{j}", 1))
                        row.append(val)
                    except ValueError:
                        return render_template('ahp.html', data='weights', error="Vse vrednosti morajo biti števila!")
                else:
                    row.append(1 / pairwise_data[j][i])
            pairwise_data.append(row)

        pairwise_matrix = np.array(pairwise_data)

        # Izračun uteži s pomočjo AHP algoritma
        column_sum = pairwise_matrix.sum(axis=0)
        normalized_matrix = pairwise_matrix / column_sum
        weights = normalized_matrix.mean(axis=1)

        # 4. Korak: Izračun rezultatov
        selected_companies = session.get('selected_companies', [])
        decision_matrix = []
        for company in selected_companies:
            row = []
            for criterion in selected_criteria:
                # Tukaj spremenimo, da iščemo ključe v manjših črkah (npr. 'prihodek', 'dobicek')
                value = company.get(criterion.lower(), 0)
                row.append(float(value.replace('$', '').replace(',', '')) if isinstance(value, str) else float(value))
            decision_matrix.append(row)

        decision_matrix = np.array(decision_matrix)
        norm_matrix = decision_matrix / decision_matrix.sum(axis=0)
        scores = norm_matrix @ weights

        # Dodeli rezultate in razvrsti
        for i, company in enumerate(selected_companies):
            company['score'] = scores[i]
        sorted_companies = sorted(selected_companies, key=lambda x: x['score'], reverse=True)

        return render_template('ahp.html', data='results', companies=sorted_companies)

if __name__ == "__main__":
    app.run(debug=True)
