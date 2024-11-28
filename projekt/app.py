from flask import Flask, render_template, request, jsonify, session
from scraping import scrapeData
import mysql.connector
import json
import os
import numpy as np
from dotenv import load_dotenv

#Nalaganje okolijskih spremenljivk iz .env datoteke
load_dotenv()

app = Flask(__name__)
app.secret_key = 'super_secret_key'

#Konfiguracija za povezavo z MySQL bazo
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
    companies = fetch_companies_from_db()
    return render_template("index.html", companies=companies)

def fetch_companies_from_db():
    """Poveže se z bazo in pridobi vse podatke o podjetjih."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM Companies")
    companies = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return companies

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

#Nalaganje podatkov iz JSON datoteke
def load_data():
    with open('fortune500.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/ahp', methods=['GET', 'POST'])
def ahp():
    companies = load_data()

    #Prvi korak: izbira podjetij
    if request.method == 'GET' or request.form.get('data') == 'selection':
        return render_template('ahp.html', data='selection', companies=companies)

    #Drugi korak: vnos uteži
    if request.form.get('data') == 'weights':
        selected_companies = get_selected_companies(companies)
        session['selected_companies'] = selected_companies
        return render_template('ahp.html', data='weights', companies=selected_companies)

    #Tretji korak: rezultati
    if request.form.get('data') == 'results':
        selected_companies = session.get('selected_companies', [])
        pairwise_matrix = get_pairwise_matrix(request)
        weights = calculate_weights(pairwise_matrix)

        decision_matrix = get_decision_matrix(selected_companies)
        norm_matrix = normalize_decision_matrix(decision_matrix)
        scores = calculate_scores(norm_matrix, weights)

        sorted_companies = sort_companies_by_score(selected_companies, scores)
        return render_template('ahp.html', data='results', companies=sorted_companies)

    return "Nepričakovano stanje!", 400

#Vrne seznam izbranih podjetij na podlagi uporabniške izbire.
def get_selected_companies(companies):
    selected_ids = request.form.getlist('selected_companies')
    return [c for c in companies if c['uvoz'] in selected_ids]

#Prebere in ustvari matriko parnih primerjav iz obrazca.
def get_pairwise_matrix(request):
    num_criteria = 4
    pairwise_data = []

    for i in range(num_criteria):
        row = []
        for j in range(num_criteria):
            if i == j:
                row.append(1.0)
            elif i < j:
                val = float(request.form.get(f"pairwise_{i}_{j}", 1))
                row.append(val)
            else:
                row.append(1 / pairwise_data[j][i])
        pairwise_data.append(row)
    
    return np.array(pairwise_data)

#Izračuna uteži na podlagi matrike parnih primerjav.
def calculate_weights(pairwise_matrix):
    column_sum = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sum
    return normalized_matrix.mean(axis=1)

#Ustvari matriko odločanja iz izbranih podjetij.
def get_decision_matrix(companies):
    decision_matrix = []
    for company in companies:
        decision_matrix.append([
            float(company['prihodek'].replace('$', '').replace(',', '')),
            float(company['dobicek'].replace('$', '').replace(',', '')),
            float(company['sredstva'].replace('$', '').replace(',', '')),
            float(company['zaposleni'].replace(',', ''))
        ])
    return np.array(decision_matrix)

#Normalizira matriko odločanja.
def normalize_decision_matrix(decision_matrix):
    return decision_matrix / decision_matrix.sum(axis=0)

#Izračuna končne rezultate na podlagi normalizirane matrike in uteži.
def calculate_scores(norm_matrix, weights):
    return norm_matrix @ weights

#Razvrsti podjetja na podlagi njihovih rezultatov.
def sort_companies_by_score(companies, scores):
    for i, company in enumerate(companies):
        company['score'] = scores[i]
    return sorted(companies, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    app.run(debug=True)
