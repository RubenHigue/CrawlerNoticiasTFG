import asyncio
import re

import csv
import sys
import uuid
from datetime import datetime

from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

from ragas import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithoutReference

import ollama

from Crawler import crawl_website
from BaseDeDatos.Cassandra import Cassandra
from BaseDeDatos.ChromaDB import ChromaDB
from sentence_transformers import SentenceTransformer

from Interface.RAGDefensaApp import RAGDefensaApp
from Ragas.BaseLLMOllama import BaseLLMOllama
import yaml

load_dotenv()

# Carga de los datos de config
with open("config_data.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Inicializar bases de datos
cassandra = Cassandra(hosts=[data.get("cassandra_host")], keyspace=data.get("keyspace"))
chroma = ChromaDB(collection_name="noticias_articulos")

# Inicializar el modelo para embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# embedding_model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")


# Función para generar respuestas con Ollama
def generate_response_with_ollama(question, context):
    prompt = (data.get("military_prompt", " ") +
              "\n\nContexto:\n" + context + "\n\nPregunta:\n" + question
              )

    prompt2 = (data.get("assistant_prompt", " ") +
               "\n\nContexto:\n" + context + "\n\nPregunta:\n" + question
               )

    response = ollama.chat(
        model=data.get("llm_model"),
        messages=[{'role': 'user', 'content': prompt}]
    )
    response2 = ollama.chat(
        model=data.get("llm_model"),
        messages=[{'role': 'user', 'content': prompt2}]
    )
    print(response.get('message', "No response received"))
    print(response2.get('message', "No response received."))
    return response.get('message', "No response received.")


# Funcion para consultas sin fechas
def response_without_dates(question):
    query_embedding = embedding_model.encode(question).tolist()
    relevant_news = chroma.search(query_embedding=query_embedding, top_k=data.get("retrieved_docs"))
    context = process_relevant_news(relevant_news)
    answer = generate_response_with_ollama(question, str(context))
    return answer


# Funcion para consultas con fechas
def response_with_dates(question, date, date2):
    if date != date2:
        query_embedding = embedding_model.encode(question).tolist()
        relevant_news = chroma.searchByRangesDate(query_embedding=query_embedding, date=date, date2=date2,
                                                  top_k=data.get("retrieved_docs"))
        context = process_relevant_news(relevant_news)
        answer = generate_response_with_ollama(question, str(context))
        return answer
    else:
        query_embedding = embedding_model.encode(question).tolist()
        relevant_news = chroma.searchByDate(query_embedding=query_embedding, date=date,
                                            top_k=data.get("retrieved_docs"))
        context = process_relevant_news(relevant_news)
        answer = generate_response_with_ollama(question, str(context))
        return answer


# Funcion para procesar el contexto y sus metadatos para el modelo
def process_relevant_news(relevant_news):
    raw_context = relevant_news.get("documents")
    metadata = relevant_news.get("metadatas")
    context = []
    for news, meta in zip(raw_context[0], metadata[0]):
        context.append("La fecha del artículo es: " + str(meta['fecha']) + " " + str(news))
    return context


# Función para introducir los articulos en Cassandra
def process_article(article_data):
    titular_modificado = article_data["titular"].replace("'", "''")

    existing_article = cassandra.query_data(
        table=data.get("table_name"),
        where_conditions={"titular": titular_modificado},
        fields=["titular"]
    )

    if existing_article:
        print(f"El artículo con el titular '{article_data['titular']}' ya existe. Saltando...")
    else:
        print(f"Insertando nuevo artículo: {article_data['titular']}")

        cassandra.insert_data(
            table=data.get("table_name"),
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


# Función para migrar los articulos de Cassandra a Chroma
def migrate_cassandra_to_chroma():
    all_articles = cassandra.get_all_entities(
        table=data.get("table_name")
    )

    print(f"Se encontraron {len(all_articles)} artículos en Cassandra para migrar a Chroma.")

    for article in all_articles:

        if not chroma.exists_by_title(article["titular"]):
            print(f"Migrando artículo con ID: {article['id']} - Titular: {article['titular']}")

            embedding = embedding_model.encode(article["noticia"])
            if article["fecha"] != "Fecha no encontrada":
                chroma.insert_article(
                    doc_id=str(article["id"]),
                    metadata={
                        "fuente": article["fuente"],
                        "fecha": datetime.strptime(article["fecha"], "%d/%m/%Y").timestamp(),
                        "titular": article["titular"],
                        "url": article["url"]
                    },
                    content=article["noticia"],
                    embedding=embedding
                )
        else:
            print(f"El articulo con titular: {article['titular']} ya está en la base de datos ChromaDB.")
    print("Migración completa. Todos los artículos han sido insertados en Chroma.")


# Funcion para procesar el archivo con las noticias del Scrapper para introducirlas en Cassandra
def process_csv_file(csv_file_path):
    print("Procesando el archivo CSV con las noticias nuevas.")
    csv.field_size_limit(2 ** 30)
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


# Funcion que ejecuta el crawler
def crawl_and_store(urls):
    for url in urls:
        print(f"Crawling {url}...")

        # Extraer solo el nombre del dominio sin "https://www."
        domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]

        filename = f"csv/noticias_defensa_{domain}.csv"
        crawl_website(url, filename)
        process_csv_file(filename)


def get_articles_from_db(table):
    print(f"Getting articles from {table}...")
    articles = []
    articles = cassandra.get_all_entities(table)
    return articles


# Funcion que ejecuta la evaluacion del proyecto
async def test_evaluation():
    user_input = "Que modelo quiere el ejercito del aire para sustituir a los F18"
    query_embedding = embedding_model.encode(user_input).tolist()
    relevant_news = chroma.search(query_embedding=query_embedding, top_k=data.get("retrieved_docs"))
    raw_context = (relevant_news.get("documents"))
    metadata = relevant_news.get("metadatas")
    context = []
    for news, meta in zip(raw_context[0], metadata[0]):
        context.append("La fecha del articulo es: " + str(meta['fecha']) + " " + str(news))
    answer = generate_response_with_ollama(user_input, str(context))
    print(context)
    answer = answer.get('content')
    print(answer)

    '''
    dataset = [
        {
            "user_input": user_input,
            "response": "El modelo que el Ejército del Aire está contemplando para sustituir a los F-18 es el F-35.",
            "reference": answer,
            "retrieved_contexts": context,
        }
    ]
    
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    llm = ChatOllama(model="gemma2")
    embeddings = OllamaEmbeddings(model="gemma2")
    #evaluator_llm = LangchainLLMWrapper(llm)
    run_config=RunConfig(timeout=6000)
    result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall()], llm=llm, embeddings=embeddings,run_config=run_config)
    print(result)
    '''

    sample = SingleTurnSample(
        user_input=user_input,
        response="El modelo que el Ejército del Aire está contemplando para sustituir a los F-18 es el F-35.",
        reference=answer,
        retrieved_contexts=context,
    )

    sample2 = SingleTurnSample(
        user_input=user_input,
        response=answer,
        reference="El modelo que el Ejército del Aire está contemplando para sustituir a los F-18 es el F-35.",
        retrieved_contexts=context,
    )

    run_config = RunConfig(max_retries=3)

    evaluator_llm = BaseLLMOllama(model_name=data.get("judge_model"), run_config=run_config)

    context_recall = LLMContextRecall(llm=evaluator_llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    print("Context Recall Results: ")
    await (context_recall.single_turn_ascore(sample))
    print("Context Precision Results: ")
    await context_precision.single_turn_ascore(sample2)


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        migrate_cassandra_to_chroma()
        if data.get("execution_mode") == "production":
            app = QApplication(sys.argv)
            window = RAGDefensaApp(response_without_dates, response_with_dates, crawl_and_store,
                                   migrate_cassandra_to_chroma, test_evaluation, get_articles_from_db)
            window.show()
            sys.exit(app.exec())
        elif data.get("execution_mode") == "test":
            print("Evaluando base de datos...")
            asyncio.run(test_evaluation())
    finally:
        cassandra.close()
