import csv
import uuid
from Crawler import crawl_website
from BaseDeDatos.Cassandra import Cassandra
from BaseDeDatos.ChromaDB import ChromaDB
from sentence_transformers import SentenceTransformer

# Inicializar bases de datos
cassandra = Cassandra(hosts=["127.0.0.1"], keyspace="noticias")
chroma = ChromaDB(collection_name="noticias_articulos")

# Inicializar el modelo para embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def process_article(article_data):
    """
    Procesa un artículo y lo almacena en Cassandra y Chroma.
    :param article_data: Diccionario con los datos del artículo.
    """
    # 1. Insertar datos tabulares en Cassandra
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
            "url": article_data["url"]
        }
    )

    # 2. Generar embedding del artículo completo
    embedding = embedding_model.encode(article_data["noticia"])

    # 3. Insertar artículo y embedding en Chroma
    chroma.insert_article(
        doc_id=str(uuid.uuid4()),
        metadata={
            "fuente": article_data["fuente"],
            "fecha": article_data["fecha"],
            "titular": article_data["titular"],
            "url": article_data["url"]
        },
        content=article_data["noticia"],
        embedding=embedding
    )


def process_csv_file(csv_file_path):
    """
    Lee un archivo CSV y procesa cada fila como un artículo.
    :param csv_file_path: Ruta al archivo CSV.
    """
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
    """
    Ejecuta el crawler y almacena los resultados en Cassandra y Chroma.
    :param url: URL del sitio web a scrapear.
    """
    crawl_website(url, "noticias_defensa.csv")
    process_csv_file("noticias_defensa.csv")


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        crawl_and_store("https://www.libertaddigital.com/defensa/")
    finally:
        cassandra.close()
