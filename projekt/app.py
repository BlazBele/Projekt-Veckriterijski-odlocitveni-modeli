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

    # Prvi korak: izbira podjetij
    if request.method == 'GET' or request.form.get('data') == 'selection':
        return render_template('ahp.html', data='selection', companies=companies)

    # Drugi korak: vnos uteži
    if request.form.get('data') == 'weights':
        selected_ids = request.form.getlist('selected_companies')
        selected_companies = [c for c in companies if c['uvoz'] in selected_ids]
        session['selected_companies'] = selected_companies
        return render_template('ahp.html', data='weights', companies=selected_companies)

    # Tretji korak: rezultati
    if request.form.get('data') == 'results':
        num_criteria = 4  # Število kriterijev (Prihodek, Dobiček, Sredstva, Zaposleni)

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
                        return render_template('ahp.html', data='weights',
                                            error="Vse vrednosti morajo biti števila!",
                                            companies=session.get('selected_companies', []))
                else:
                    # Spodnji trikotnik je inverz zgornjega
                    row.append(1 / pairwise_data[j][i])
            pairwise_data.append(row)

        pairwise_matrix = np.array(pairwise_data)
        print(f"Matrika parnih primerjav:\n{pairwise_matrix}")

        # Izračun uteži s pomočjo AHP algoritma
        column_sum = pairwise_matrix.sum(axis=0)
        normalized_matrix = pairwise_matrix / column_sum
        weights = normalized_matrix.mean(axis=1)
        print(f"Uteži kriterijev: {weights}")

        # Odločitvena matrika podjetij
        selected_companies = session.get('selected_companies', [])
        decision_matrix = []
        for company in selected_companies:
            try:
                decision_matrix.append([
                    float(company['prihodek'].replace('$', '').replace(',', '')),
                    float(company['dobicek'].replace('$', '').replace(',', '')),
                    float(company['sredstva'].replace('$', '').replace(',', '')),
                    float(company['zaposleni'].replace(',', ''))
                ])
            except ValueError:
                return render_template('ahp.html', data='weights',
                                    error=f"Napaka pri podatkih podjetja {company['podjetje']}",
                                    companies=selected_companies)

        decision_matrix = np.array(decision_matrix)
        print(f"Matrika odločanja: {decision_matrix}")

        # Normalizacija matrike odločanja
        norm_matrix = decision_matrix / decision_matrix.sum(axis=0)
        print(f"Normalizirana matrika: {norm_matrix}")

        # Matrično množenje za ocene
        scores = norm_matrix @ weights
        print(f"Rezultati: {scores}")

        # Dodeli rezultate in razvrsti
        for i, company in enumerate(selected_companies):
            company['score'] = scores[i]
        sorted_companies = sorted(selected_companies, key=lambda x: x['score'], reverse=True)

        return render_template('ahp.html', data='results', companies=sorted_companies)

    return "Nepričakovano stanje!", 400



if __name__ == "__main__":
    app.run(debug=True)
