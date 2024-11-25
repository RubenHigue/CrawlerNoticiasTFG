from Crawler import crawl_website

if __name__ == "__main__":
    URL = "https://www.libertaddigital.com/defensa/"
    OUTPUT_FILE = "noticias_defensa.csv"

    print(f"Iniciando crawler para: {URL}")
    crawl_website(URL, OUTPUT_FILE)
    print(f"Crawler finalizado. Datos guardados en {OUTPUT_FILE}")
