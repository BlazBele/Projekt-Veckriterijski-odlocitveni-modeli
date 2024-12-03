from bson import ObjectId
from flask import Flask, render_template, request, jsonify, session
from scraping import scrapeData
from pymongo import MongoClient
import json
import os
import numpy as np
from dotenv import load_dotenv

# Nalaganje okolijskih spremenljivk iz .env datoteke
load_dotenv()

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Konfiguracija za povezavo z MongoDB
client = MongoClient(os.getenv('MONGO_URI'))
db = client[os.getenv('DB_NAME')]

@app.route("/")
def index():
    """Prikaže glavno stran s tabelo podjetij iz MongoDB."""
    companies = fetch_companies_from_db()
    return render_template("index.html", companies=companies)

def fetch_companies_from_db():
    """Poveže se z MongoDB in pridobi vse podatke o podjetjih."""
    companies = list(db.Companies.find({}, {"_id": 0}))  # Izključimo "_id", če ni potreben
    return companies

def convert_objectid_to_str(data):
    if isinstance(data, list):  # Če je seznam, obdelamo vsak element
        for item in data:
            convert_objectid_to_str(item)
    elif isinstance(data, dict):  # Če je slovar, preverimo vsako vrednost
        for key, value in data.items():
            if isinstance(value, ObjectId):  # Če je vrednost ObjectId, jo pretvorimo v niz
                data[key] = str(value)
            elif isinstance(value, (dict, list)):  # Če je vrednost slovar ali seznam, rekurzivno kličemo funkcijo
                convert_objectid_to_str(value)
    return data

@app.route("/scrape", methods=["POST"])
def scrape():
    """API endpoint za scrapanje podatkov in shranjevanje v MongoDB."""
    url = request.json.get("url")
    if not url:
        return jsonify({"status": "error", "message": "Manjka URL"}), 400

    data, message = scrapeData(url)
    if data is None:
        return jsonify({"status": "error", "message": message}), 500

    # Odstranimo _id iz podatkov pred shranjevanjem
    for item in data:
        if '_id' in item:
            del item['_id']  # Odstranimo obstoječi _id, da MongoDB sam dodeli nov

    # Shranimo podatke v MongoDB
    save_to_db(data)

    # Preden vrnemo podatke, pretvorimo ObjectId v nize
    data = convert_objectid_to_str(data)

    return jsonify({"status": "success", "message": message, "data": data}), 200


import json  # Za serializacijo podatkov

@app.route("/wsm", methods=["GET", "POST"])
def wsm():
    """Stran za metodo WSM in obdelavo rezultatov analize."""
    companies = fetch_companies_from_db()
    data = 'selection'  # Določa korak (selection, weights, results)
    results = None
    selected_companies = []

    if request.method == "POST":
        if request.form.get('data') == 'selection':  # Prvi korak: Izbira podjetij
            selected_ids = request.form.getlist('selected_companies')
            if not selected_ids:
                error = "Izberite vsaj eno podjetje."
                return render_template("wsm.html", companies=companies, data='selection', error=error)

            # Shrani izbrana podjetja za naslednji korak
            selected_companies = [c for c in companies if c['uvoz'] in selected_ids]
            return render_template("wsm.html", selected_companies=json.dumps(selected_companies), data='weights')

        elif request.form.get('data') == 'weights':  # Drugi korak: Vnos uteži
            selected_companies = json.loads(request.form.get('selected_companies_json', '[]'))

            # Pridobitev uteži iz obrazca
            weight_income = float(request.form['weight_income'])
            weight_profit = float(request.form['weight_profit'])
            weight_assets = float(request.form['weight_assets'])
            weight_employees = float(request.form['weight_employees'])

            # Izračun ocen na podlagi WSM
            for company in selected_companies:
                income = float(company['prihodek'].replace('$', '').replace(',', ''))
                profit = float(company['dobicek'].replace('$', '').replace(',', ''))
                assets = float(company['sredstva'].replace('$', '').replace(',', ''))
                employees = float(company['zaposleni'].replace(',', ''))

                # Izračun WSM ocene
                score = (weight_income * income +
                         weight_profit * profit +
                         weight_assets * assets +
                         weight_employees * employees)
                
                company['score'] = round(score, 1)

            # Razvrščanje podjetij po WSM oceni
            results = sorted(selected_companies, key=lambda x: x['score'], reverse=True)
            data = 'results'

    return render_template("wsm.html", companies=companies, data=data, selected_companies=json.dumps(selected_companies), results=results)



















def save_to_db(data):
    """Shrani podatke v MongoDB."""
    db.Companies.insert_many(data)

@app.route("/topsis")
def topsis():
    """Stran za metodo TOPSIS."""
    return render_template("topsis.html")

@app.route("/promethee")
def promethee():
    """Stran za metodo PROMETHEE."""
    return render_template("promethee.html")

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
    companies = fetch_companies_from_db()

    # Prvi korak: izbira podjetij
    if request.method == 'GET' or request.form.get('data') == 'selection':
        return render_template('ahp.html', data='selection', companies=companies)

    # Drugi korak: vnos uteži
    if request.form.get('data') == 'weights':
        selected_companies = get_selected_companies(companies)
        session['selected_companies'] = selected_companies
        return render_template('ahp.html', data='weights', companies=selected_companies)

    # Tretji korak: rezultati
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

# Vrne seznam izbranih podjetij na podlagi uporabniške izbire.
def get_selected_companies(companies):
    selected_ids = request.form.getlist('selected_companies')
    return [c for c in companies if c['uvoz'] in selected_ids]

# Prebere in ustvari matriko parnih primerjav iz obrazca.
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

# Izračuna uteži na podlagi matrike parnih primerjav.
def calculate_weights(pairwise_matrix):
    column_sum = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sum
    return normalized_matrix.mean(axis=1)

# Ustvari matriko odločanja iz izbranih podjetij.
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

# Normalizira matriko odločanja.
def normalize_decision_matrix(decision_matrix):
    return decision_matrix / decision_matrix.sum(axis=0)

# Izračuna končne rezultate na podlagi normalizirane matrike in uteži.
def calculate_scores(norm_matrix, weights):
    return norm_matrix @ weights

# Razvrsti podjetja na podlagi njihovih rezultatov.
def sort_companies_by_score(companies, scores):
    for i, company in enumerate(companies):
        company['score'] = scores[i]
    return sorted(companies, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    app.run(debug=True)
