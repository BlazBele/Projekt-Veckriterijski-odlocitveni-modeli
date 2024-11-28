import mysql.connector
from mysql.connector import Error

def test_mysql_connection(host, port, user, password):
    """Testira povezavo do MySQL strežnika brez navedbe baze."""
    connection = None  # Inicializacija spremenljivke
    try:
        # Vzpostavitev povezave
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )

        if connection.is_connected():
            print("Povezava z MySQL strežnikom je uspešna.")

            # Izvedba poizvedbe
            cursor = connection.cursor()
            cursor.execute("SHOW DATABASES;")
            databases = cursor.fetchall()
            print("Seznam baz:", databases)

    except Error as e:
        print("Napaka pri povezavi z bazo:", e)

    finally:
        # Zapri povezavo, če je bila vzpostavljena
        if connection is not None and connection.is_connected():
            cursor.close()
            connection.close()
            print("Povezava z MySQL strežnikom je zaprta.")

# Nastavitve za povezavo
if __name__ == "__main__":
    host = 'database-1.c96m86o06xrt.eu-north-1.rds.amazonaws.com'           # Zamenjajte s svojim AWS MySQL gostiteljem
    port = 3306                   # Privzeti port za MySQL
    user = 'admin'  # Zamenjajte s svojim uporabniškim imenom
    password = 'adminadmin'     # Zamenjajte s svojim geslom

    # Klic funkcije za testiranje povezave
    test_mysql_connection(host, port, user, password)
