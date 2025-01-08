def ahp():
    """Stran za metodo AHP in obdelavo rezultatov analize."""
    # Fetch podatkov za analizo
    companies = fetch_companies_from_db()  # Pridobitev podatkov iz baze
    criteria = ["prihodek", "dobicek", "sredstva", "zaposleni"]  # Merila za analizo
    data = 'selection'
    results = []

    # Ko uporabnik izbere podjetja za analizo
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

            selected_companies = [c for c in companies if c['uvoz'] in selected_ids]
            
            # Shrani izbrane podjetje v sejo
            session['selected_companies'] = selected_companies
        
            return render_template(
                "ahp.html",
                selected_companies=selected_companies,
                data='pairwise',
                criteria=criteria
            )
        
        if request.form.get('data') == 'pairwise':  # Drugi korak: Vnos parnih primerjav
            pairwise_criteria = []       

            # Prenesi izbrane podjetja iz seje
            selected_companies = session.get('selected_companies', [])

            # Če ni podjetij, jih preusmeri nazaj na prvi korak
            if not selected_companies:
                error = "Izberite podjetja pred nadaljevanjem."
                return render_template(
                    "ahp.html",
                    companies=companies,
                    data='selection',
                    error=error
                )

            # Zbiranje parnih primerjav za kriterije
            for i in range(len(criteria)):
                for j in range(i + 1, len(criteria)):
                    pairwise_criteria.append(float(request.form[f'pairwise_{i}_{j}']))

            # Zbiranje parnih primerjav za vsak kriterij
            pairwise_prihodek = []
            pairwise_dobicek = []
            pairwise_sredstva = []
            pairwise_zaposleni = []

            # Dopolnitev celotne matrike (dodajanje obratnih vrednosti)
            def fill_full_matrix(upper_triangle, n):
                matrix = np.eye(n)  # Začetna matrika z enicami na diagonali
                index = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if index < len(upper_triangle):
                            matrix[i][j] = upper_triangle[index]
                            matrix[j][i] = 1 / matrix[i][j]  # Dopolni spodnji del matrike z obratnimi vrednostmi
                            index += 1
                return matrix
            
            # Zbiranje zgornjega desnega trikotnika
            for i in range(len(selected_companies)):
                for j in range(i + 1, len(selected_companies)):
                    # Zberemo vrednosti za posamezen kriterij
                    prihodek = request.form.get(f'pairwise_prihodek_{i}_{j}')
                    dobiček = request.form.get(f'pairwise_dobicek_{i}_{j}')
                    sredstva = request.form.get(f'pairwise_sredstva_{i}_{j}')
                    zaposleni = request.form.get(f'pairwise_zaposleni_{i}_{j}')

                    pairwise_prihodek.append(float(prihodek))
                    pairwise_dobicek.append(float(dobiček))
                    pairwise_sredstva.append(float(sredstva))
                    pairwise_zaposleni.append(float(zaposleni))

            # Dimenzije matrik (število podjetij)
            n_companies = len(selected_companies)

            # Izračun celotnih matrik za vse kriterije
            full_prihodek_matrix  = (fill_full_matrix(pairwise_prihodek, n_companies))
            full_dobicek_matrix = (fill_full_matrix(pairwise_dobicek, n_companies))
            full_sredstva_matrix = (fill_full_matrix(pairwise_sredstva, n_companies))
            full_zaposleni_matrix = (fill_full_matrix(pairwise_zaposleni, n_companies))

            reshaped_full_prihodek_matrix = np.array(full_prihodek_matrix.reshape(n_companies, n_companies))
            reshaped_full_dobicek_matrix = np.array(full_dobicek_matrix.reshape(n_companies, n_companies))
            reshaped_sredstva_matrix = np.array(full_sredstva_matrix.reshape(n_companies, n_companies))
            reshaped_zaposleni_matrix = np.array(full_zaposleni_matrix.reshape(n_companies, n_companies))

            stacked_matrices = np.vstack((
                reshaped_full_prihodek_matrix,
                reshaped_full_dobicek_matrix,
                reshaped_sredstva_matrix,
                reshaped_zaposleni_matrix
            ))

            print(stacked_matrices)

            # Dopolni matriko s podatki iz zgornjega trikotnika
            full_PCM = fill_full_matrix(pairwise_criteria, len(criteria))

            print(full_PCM)


            # Izračun prioritetnih vektorjev z metodo AHP
            steviloAlternativ = len(selected_companies)  # Število kriterijev
            steviloKriterijev = len(criteria)  # Število podjetij

    

            print("steviloAlternativ", steviloAlternativ)
            print("Število kriterijev", steviloKriterijev)


            # Tukaj bi bil tvoj AHP algoritem
            results = ahpCalc(stacked_matrices, full_PCM, steviloAlternativ, steviloKriterijev, c=2)

            print(results)

            # Pripravi rezultate za prikaz
            final_results = []
            for i, score in enumerate(results):
                final_results.append({
                    'podjetje': selected_companies[i]['podjetje'],  # Poveži ime podjetja
                    'prihodek': convert_to_numeric(selected_companies[i]['prihodek']),  # Pretvori prihodek v številko
                    'dobicek': convert_to_numeric(selected_companies[i]['dobicek']),  # Pretvori dobiček v številko
                    'sredstva': convert_to_numeric(selected_companies[i]['sredstva']),  # Pretvori sredstva v številko
                    'zaposleni': convert_to_numeric(selected_companies[i]['zaposleni']),  # Pretvori število zaposlenih v številko
                    'score': score  # Dodaj rezultat AHP
                })

            return render_template("ahp.html", results=final_results, data='results', selected_companies=selected_companies)


    return render_template("ahp.html", companies=companies, data=data)

def convert_to_numeric(value):
    """Pretvori vrednosti, ki so v obliki besedila z simboli, v številske vrednosti."""
    value = value.replace(',', '')  # Odstrani vejice
    value = value.replace('$', '')  # Odstrani znak za dolar
    try:
        return float(value)  # Pretvori v float
    except ValueError:
        return 0  # Če ni mogoče pretvoriti, vrni 0

def norm(x):
    """ x is the pairwise comparison matrix for the 
    criteria or the alternatives
    """
    k = array(sum(x, 0))
    z = array([[round(x[i, j] / k[j], 3) 
        for j in range(x.shape[1])]
        for i in range(x.shape[0])])
    return z

# geometric mean method
def geomean(x):
    """ x is the pairwise comparison matrix for the
    criteria or the alternatives
    """
    z = [1] * x.shape[0]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] = z[i] * x[i][j]
        z[i] = pow(z[i], (1 / x.shape[0]))
    return z

# AHP method: it calls the other functions
def ahpCalc(PCM, PCcriteria, m, n, c):
    """ PCM is the pairwise comparison matrix for the
    alternatives,  PCcriteria is the pairwise comparison 
    matrix for the criteria, m is the number of the 
    alternatives, n is the number of the criteria, and 
    c is the method to estimate a priority vector (1 for 
    eigenvector, 2 for normalized column sum, and 3 for
    geometric mean)
    """
    # calculate the priority vector of criteria
    if c == 1: # eigenvector
        val, vec = sc.eigs(PCcriteria, k = 1, which = 'LM')
        eigcriteria = real(vec)
        w = eigcriteria / sum(eigcriteria)
        w = array(w).ravel()
    elif c == 2: # normalized column sum
        normPCcriteria = norm(PCcriteria)
        w = array(sum(normPCcriteria, 1) / n)
    else: # geometric mean
        GMcriteria = geomean(PCcriteria)
        w = GMcriteria / sum(GMcriteria)
    # calculate the local priority vectors for the 
    # alternatives
    S = []
    for i in range(n):
        if c == 1: # eigenvector
            val, vec = sc.eigs(PCM[i * m:i * m + m, 0:m],
                k = 1, which = 'LM')
            eigalter = real(vec)
            s = eigalter / sum(eigalter)
            s = array(s).ravel()
        elif c == 2: # normalized column sum
            normPCM = norm(PCM[i*m:i*m+m,0:m])
            s = array(sum(normPCM, 1) / m)
        else: # geometric mean
            GMalternatives = geomean(PCM[i*m:i*m+m,0:m])
            s = GMalternatives / sum(GMalternatives)
        S.append(s)
    S = transpose(S)

    # calculate the global priority vector for the
    # alternatives
    v = S.dot(w.T)

    return v