from bson import ObjectId
from flask import Flask, render_template, request, jsonify, session
from scraping import scrapeData
from pymongo import MongoClient
import json
import os
import numpy as np
from dotenv import load_dotenv
from pyDecision.algorithm import ahp_method
from pyDecision.algorithm import topsis_method
import matplotlib.pyplot as plt
import io
import base64

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


@app.route("/ahp", methods=["GET", "POST"])
def ahp():
    """Stran za metodo AHP in obdelavo rezultatov analize."""
    # Fetch podatkov za analizo
    companies = fetch_companies_from_db()
    criteria = ["prihodek", "dobicek", "sredstva", "zaposleni"]  # Merila za analizo v manjših črkah
    data = 'selection'
    results = None
    selected_companies = []
    error = None  # Inicializiraj napako kot None
    graph_url = None  # Inicializiraj URL grafa za prikaz
    consistency_ratio = None  # Inicializiraj konsistentnost matrike (RC)

    if request.method == "POST":
        if request.form.get('data') == 'selection':  # Prvi korak: Izbor podjetij
            selected_ids = request.form.getlist('selected_companies')
            if not selected_ids:
                error = "Izberite vsaj eno podjetje."
                return render_template(
                    "ahp.html",
                    companies=companies,
                    data='selection',
                    error=error
                )

            selected_companies = [
                c for c in companies if c['uvoz'] in selected_ids
            ]
            return render_template(
                "ahp.html",
                selected_companies=json.dumps(selected_companies),
                data='pairwise',
                criteria=criteria
            )

        elif request.form.get('data') == 'pairwise':  # Drugi korak: Vnos parnih primerjav
            # Deserialize selected companies
            selected_companies = json.loads(request.form.get('selected_companies_json', '[]'))

            # Zbiranje matrike parnih primerjav
            pairwise_data = []
            for i in range(len(criteria)):
                row = []
                for j in range(len(criteria)):
                    if j > i:
                        # Vrednosti parne primerjave (zgornja trikotna matrika)
                        value = float(request.form.get(f'pairwise_{i}_{j}', '1'))
                        row.append(value)
                    elif j == i:
                        row.append(1)  # Diagonalna vrednost je vedno 1
                    else:
                        # Inverzna vrednost za spodnjo trikotno matriko
                        row.append(1 / pairwise_data[j][i])
                pairwise_data.append(row)

            pairwise_matrix = np.array(pairwise_data)

            # Izračun uteži in konsistentnosti
            weights, rc = ahp_method(pairwise_matrix, wd='geometric')

            consistency_ratio = round(rc,2)  # Shrani konsistentnost matrike
            
            if rc > 0.1:
                error = "Razmerja so nekonsistentna (RC > 0.1). Poskusite znova."
                return render_template(
                    "ahp.html",
                    selected_companies=json.dumps(selected_companies),
                    data='pairwise',
                    criteria=criteria,
                    error=error  # Napaka se zdaj prikaže v šabloni
                )

            # Izračun AHP ocen za podjetja
            for company in selected_companies:
                # Pretvarjanje vrednosti v številke (opazite, da zdaj dostopamo do majhnih ključev)
                scores = [
                    float(company[crit].replace('$', '').replace(',', ''))  # Uporabimo male črke za ključe
                    for crit in criteria
                ]
                # Utežena vsota
                company['score'] = round(sum(w * s for w, s in zip(weights, scores)), 1)

            # Razvrstitev rezultatov po AHP oceni
            results = sorted(
                selected_companies,
                key=lambda x: x['score'],
                reverse=True
            )
            data = 'results'

            # Generiranje grafa za razvrstitev podjetij
            plt.figure(figsize=(10, 6))
            company_names = [c['podjetje'] for c in results]
            scores = [c['score'] for c in results]
            plt.barh(company_names, scores, color='orange')
            plt.title('Razvrstitev podjetij (AHP)')
            plt.xlabel('AHP Ocena')
            plt.ylabel('Podjetja')
            plt.gca().invert_yaxis()

            # Pretvorba grafa v Base64 string
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

    return render_template(
        "ahp.html",
        companies=companies,
        data=data,
        selected_companies=json.dumps(selected_companies),
        results=results,
        criteria=criteria,
        error=error,  # Posredujemo napako v šablono
        graph_url=graph_url,  # Posredujemo graf za prikaz
        consistency_ratio=consistency_ratio  # Posredujemo konsistentnost matrike (RC)
    )





@app.route("/wsm", methods=["GET", "POST"])
def wsm():
    """Stran za metodo WSM in obdelavo rezultatov analize."""
    companies = fetch_companies_from_db()
    data = 'selection'  # Določa korak (selection, weights, results)
    results = None
    selected_companies = []
    graph_url = None  #

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

            # Generiranje grafa
            plt.figure(figsize=(10, 6))
            company_names = [c['podjetje'] for c in results]
            scores = [c['score'] for c in results]
            plt.barh(company_names, scores, color='skyblue')
            plt.xlabel('WSM Ocena')
            plt.ylabel('Podjetje')
            plt.title('WSM Rezultati')
            plt.gca().invert_yaxis()

            # Pretvori graf v Base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

    return render_template("wsm.html", companies=companies, data=data, selected_companies=json.dumps(selected_companies), results=results, graph_url=graph_url)



@app.route('/topsis', methods=['GET', 'POST'])
def topsis():
    companies = fetch_companies_from_db()
    data = 'selection'  # Določa korak (selection, weights, results)
    results = None
    selected_companies = []
    graph_url = None  # Inicializacija za graf

    if request.method == 'POST':
        data = request.form.get('data')
        if data == 'selection':
            # Izbor podjetij
            selected_companies_ids = request.form.getlist('selected_companies')
            selected_companies = [company for company in companies if str(company['uvoz']) in selected_companies_ids]
            return render_template('topsis.html', data='weights', companies=companies, selected_companies=json.dumps(selected_companies))

        elif data == 'weights':
            # Pridobivanje uteži
            weight_income = float(request.form['weight_income'])
            weight_profit = float(request.form['weight_profit'])
            weight_assets = float(request.form['weight_assets'])
            weight_employees = float(request.form['weight_employees'])

            # Izbor podjetij
            selected_companies_json = request.form['selected_companies_json']
            selected_companies = json.loads(selected_companies_json)

            # Priprava podatkov za TOPSIS
            dataset = []
            for company in selected_companies:
                income = float(company['prihodek'].replace('$', '').replace(',', ''))
                profit = float(company['dobicek'].replace('$', '').replace(',', ''))
                assets = float(company['sredstva'].replace('$', '').replace(',', ''))
                employees = float(company['zaposleni'].replace(',', ''))
                dataset.append([income, profit, assets, employees])

            weights = [weight_income, weight_profit, weight_assets, weight_employees]
            criterion_type = ['max', 'max', 'max', 'max']  # Privzeto "max" za vse kriterije

            # Klic TOPSIS metode
            dataset_np = np.array(dataset)
            relative_closeness = topsis_method(dataset_np, weights, criterion_type, graph=False, verbose=False)

            # Priprava rezultatov
            for i, company in enumerate(selected_companies):
                company['score'] = round(relative_closeness[i], 2)

            results = sorted(selected_companies, key=lambda x: x['score'], reverse=True)
            data = 'results'

            # Generiranje grafa
            plt.figure(figsize=(10, 6))
            company_names = [c['podjetje'] for c in results]
            scores = [c['score'] for c in results]
            plt.barh(company_names, scores, color='lime')
            plt.xlabel('TOPSIS Ocena')
            plt.ylabel('Podjetje')
            plt.title('TOPSIS Rezultati')
            plt.gca().invert_yaxis()

            # Pretvorba grafa v Base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

    return render_template('topsis.html', data=data, companies=companies, results=results, graph_url=graph_url)



@app.route("/promethee", methods=["GET", "POST"])
def promethee():
    """Stran za metodo PROMETHEE in obdelavo rezultatov analize."""
    companies = fetch_companies_from_db()
    data = 'selection'  # Določa korak (selection, weights, results)
    results = None
    selected_companies = []
    graph_url = None  # Inicializacija za graf

    if request.method == "POST":
        if request.form.get('data') == 'selection':  # Prvi korak: Izbira podjetij
            selected_ids = request.form.getlist('selected_companies')
            if not selected_ids:
                error = "Izberite vsaj eno podjetje."
                return render_template("promethee.html", companies=companies, data='selection', error=error)

            # Shrani izbrana podjetja za naslednji korak
            selected_companies = [c for c in companies if c['uvoz'] in selected_ids]
            return render_template("promethee.html", selected_companies=json.dumps(selected_companies), data='weights')

        elif request.form.get('data') == 'weights':  # Drugi korak: Vnos uteži
            selected_companies = json.loads(request.form.get('selected_companies_json', '[]'))

            # Pridobitev uteži iz obrazca
            weight_income = float(request.form['weight_income'])
            weight_profit = float(request.form['weight_profit'])
            weight_assets = float(request.form['weight_assets'])
            weight_employees = float(request.form['weight_employees'])

            # Za PROMETHEE potrebujemo matriko preferenc in uteži
            preferences = []
            for company in selected_companies:
                income = float(company['prihodek'].replace('$', '').replace(',', ''))
                profit = float(company['dobicek'].replace('$', '').replace(',', ''))
                assets = float(company['sredstva'].replace('$', '').replace(',', ''))
                employees = float(company['zaposleni'].replace(',', ''))
                preferences.append([income, profit, assets, employees])

            # Uteži za različne dejavnike
            weights = [weight_income, weight_profit, weight_assets, weight_employees]

            # Izračun PROMETHEE rezultatov
            positive_flows, negative_flows = promethee_method(preferences, weights)

            # Združevanje rezultatov
            for i, company in enumerate(selected_companies):
                company['positive_flow'] = positive_flows[i]
                company['negative_flow'] = negative_flows[i]
                company['score'] = round(positive_flows[i] - negative_flows[i], 1)

            # Razvrščanje podjetij po rezultatu (score)
            results = sorted(selected_companies, key=lambda x: x['score'], reverse=True)
            data = 'results'

            # Generiranje grafa
            plt.figure(figsize=(10, 6))
            company_names = [c['podjetje'] for c in results]
            scores = [c['score'] for c in results]
            plt.barh(company_names, scores, color='lightgreen')
            plt.xlabel('PROMETHEE Ocena')
            plt.ylabel('Podjetje')
            plt.title('PROMETHEE Rezultati')
            plt.gca().invert_yaxis()

            # Pretvorba grafa v Base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

    return render_template("promethee.html", companies=companies, data=data, selected_companies=json.dumps(selected_companies), results=results, graph_url=graph_url)



def promethee_method(preferences, weights):
    """
    Implementacija osnovne PROMETHEE metode za izračun pozitivnih in negativnih tokov.
    preferences - seznam preferenc (vsaka vrstica je podjetje)
    weights - uteži za vsak kriterij (prihodek, dobiček, sredstva, zaposleni)
    """
    n = len(preferences)
    m = len(preferences[0])
    
    # Matrika prednosti (preference matrix)
    preference_matrix = []
    for i in range(n):
        preference_row = []
        for j in range(n):
            # Razlika med podjetjema i in j za vsak kriterij
            preference = sum([weights[k] * (preferences[i][k] - preferences[j][k])
                              for k in range(m)])
            preference_row.append(preference)
        preference_matrix.append(preference_row)

    # Izračun pozitivnih in negativnih tokov
    positive_flows = []
    negative_flows = []
    for i in range(n):
        positive_flow = 0
        negative_flow = 0
        for j in range(n):
            if preference_matrix[i][j] > 0:
                positive_flow += preference_matrix[i][j]
            else:
                negative_flow += -preference_matrix[i][j]
        positive_flows.append(positive_flow)
        negative_flows.append(negative_flow)

    return positive_flows, negative_flows


def save_to_db(data):
    """Shrani podatke v MongoDB."""
    db.Companies.insert_many(data)


@app.route("/upload")
def upload():
    """Stran za metodo Upload"""
    return render_template("upload.html")

# Nalaganje podatkov iz JSON datoteke
def load_data():
    with open('fortune500.json', 'r', encoding='utf-8') as f:
        return json.load(f)


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
