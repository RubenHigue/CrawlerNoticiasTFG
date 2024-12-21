import csv
import uuid
import sys

import ollama

from Crawler import crawl_website
from BaseDeDatos.Cassandra import Cassandra
from BaseDeDatos.ChromaDB import ChromaDB
from sentence_transformers import SentenceTransformer

# Inicializar bases de datos
cassandra = Cassandra(hosts=["127.0.0.1"], keyspace="noticias")
chroma = ChromaDB(collection_name="noticias_articulos")

# Inicializar el modelo para embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Función para generar respuestas con Ollama
def generate_response_with_ollama(question, context):
    prompt = (
            "Eres un asistente militar de defensa para tareas de respuesta a preguntas. "
            "Utiliza las siguientes piezas de contexto recuperado para responder a la pregunta. "
            "Si no sabes la respuesta, di que no la sabes. "
            "Usa un máximo de tres frases y mantén la respuesta concisa."
            "\n\nContexto:\n" + context + "\n\nPregunta:\n" + question
    )

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.get('message', "No response received.")


def main_menu():
    while True:
        print("\n=== Menú Principal ===")
        print("1. Ejecutar scraping y almacenar en bases de datos")
        print("2. Consultar datos en ChromaDB")
        print("3. Consultar a Ollama")
        print("4. Copiar los datos a ChromaDB")
        print("5. Salir")
        choice = input("Selecciona una opción (1-4): ")

        if choice == "1":
            crawl_and_store("https://www.libertaddigital.com/defensa/")
            print("Scraping y almacenamiento completados.")
        elif choice == "2":
            query = input("Introduce tu consulta: ")
            query_embedding = embedding_model.encode(query).tolist()
            results = chroma.search(query_embedding=query_embedding)
            print("\n=== Resultados de la consulta ===")
            for i, result in enumerate(results["documents"], 1):
                print(f"Resultado {i}: {result}")
        elif choice == "3":
            question = input("Introduce tu consulta: ")
            query_embedding = embedding_model.encode(question).tolist()
            relevant_news = chroma.search(query_embedding=query_embedding)
            context = (str(relevant_news.get("documents")))
            answer = generate_response_with_ollama(question, context)

            print(answer)
        elif choice == "4":
            print("Copying the data from Cassandra to local database...")
            migrate_cassandra_to_chroma()
            print("Done.")
        elif choice == "5":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción entre 1 y 5.")


def process_article(article_data):
    titular_modificado = article_data["titular"].replace("'", "''")

    existing_article = cassandra.query_data(
        table="noticias_tabulares",
        where_conditions={"titular": titular_modificado},
        fields=["titular"]
    )

    if existing_article:
        print(f"El artículo con el titular '{article_data['titular']}' ya existe. Saltando...")
    else:
        print(f"Insertando nuevo artículo: {article_data['titular']}")

        cassandra.insert_data(
            table="noticias_tabulares",
            data={
                "id": uuid.uuid4(),
                "fuente": article_data["fuente"],
                "fecha": article_data["fecha"],
                "hora": article_data["hora"],
                "titular": article_data["titular"],
                "autor": article_data["autor"],
                "autor_url": article_data["autor_url"],
                "noticia": article_data["noticia"],
                "articulo_original": article_data["articulo_original"],
                "url": article_data["url"]

            }
        )


def migrate_cassandra_to_chroma():
    all_articles = cassandra.get_all_entities(
        table="noticias_tabulares"
    )

    print(f"Se encontraron {len(all_articles)} artículos en Cassandra para migrar a Chroma.")

    for article in all_articles:
        print(f"Migrando artículo con ID: {article['id']} - Titular: {article['titular']}")

        embedding = embedding_model.encode(article["noticia"])

        chroma.insert_article(
            doc_id=str(article["id"]),
            metadata={
                "fuente": article["fuente"],
                "fecha": article["fecha"],
                "titular": article["titular"],
                "url": article["url"]
            },
            content=article["noticia"],
            embedding=embedding
        )

    print("Migración completa. Todos los artículos han sido insertados en Chroma.")


def process_csv_file(csv_file_path):
    print("Procesando el archivo CSV con las noticias nuevas.")
    csv.field_size_limit(sys.maxsize)
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            process_article({
                "fuente": row.get("fuente", ""),
                "fecha": row.get("fecha", ""),
                "hora": row.get("hora", ""),
                "titular": row.get("titular", ""),
                "autor": row.get("autor", ""),
                "autor_url": row.get("autor_url", ""),
                "noticia": row.get("noticia", ""),
                "articulo_original": row.get("articulo_original", ""),
                "url": row.get("url", "")
            })


def crawl_and_store(url):
    print(f"Crawling {url}...")
    crawl_website(url, "noticias_defensa.csv")
    process_csv_file("noticias_defensa.csv")


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        main_menu()
    finally:
        cassandra.close()
