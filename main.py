import asyncio
import csv
import uuid
import sys
from datetime import datetime
from dotenv import load_dotenv
import os

from langchain_community.llms.ollama import Ollama
from langchain_core.language_models import LLM
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas import RunConfig
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithoutReference
from langchain_community.llms import VLLM

import ollama

from Crawler import crawl_website
from BaseDeDatos.Cassandra import Cassandra
from BaseDeDatos.ChromaDB import ChromaDB
from sentence_transformers import SentenceTransformer

from Ragas.BaseLLMOllama import BaseLLMOllama
from pathlib import Path
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


def main_menu():
    while True:
        print("\n=== Menú Principal ===")
        print("1. Ejecutar scraping y almacenar en bases de datos")
        print("2. Consultar datos en ChromaDB")
        print("3. Consultar a Ollama")
        print("4. Consultar a Ollama con fecha aproximada")
        print("5. Consultar a Ollama con rangos de fechas")
        print("6. Copiar los datos a ChromaDB")
        print("7. Evaluacion de la base de datos")
        print("8. Salir")
        choice = input("Selecciona una opción (1-8): ")

        if choice == "1":
            crawl_and_store(data.get("news_url"))
            print("Scraping y almacenamiento completados.")
        elif choice == "2":
            query = input("Introduce tu consulta: ")
            query_embedding = embedding_model.encode(query).tolist()
            results = chroma.search(query_embedding=query_embedding, top_k=data.get("retrieved_docs"))
            print("\n=== Resultados de la consulta ===")
            for i, result in enumerate(results["documents"], 1):
                print(f"Resultado {i}: {result}")
        elif choice == "3":
            question = input("Introduce tu consulta: ")
            query_embedding = embedding_model.encode(question).tolist()
            relevant_news = chroma.search(query_embedding=query_embedding, top_k=data.get("retrieved_docs"))
            raw_context = (relevant_news.get("documents"))
            metadata = relevant_news.get("metadatas")
            context = []
            for news, meta in zip(raw_context[0], metadata[0]):
                context.append("La fecha del articulo es: " + str(meta['fecha']) + " " + str(news))
            answer = generate_response_with_ollama(question, str(context))
            # print(answer)
        elif choice == "4":
            question = input("Introduce tu consulta: ")
            date = input("Introduce la fecha en la que desea buscar: ")
            query_embedding = embedding_model.encode(question).tolist()
            relevant_news = chroma.searchByDate(query_embedding=query_embedding, date=date,
                                                top_k=data.get("retrieved_docs"))
            raw_context = (relevant_news.get("documents"))
            metadata = relevant_news.get("metadatas")
            context = []
            for news, meta in zip(raw_context[0], metadata[0]):
                context.append("La fecha del articulo es: " + str(meta['fecha']) + " " + str(news))
            answer = generate_response_with_ollama(question, str(context))
            # print(answer)
        elif choice == "5":
            question = input("Introduce tu consulta: ")
            date = input("Introduce la fecha en la que desea buscar: ")
            date2 = input("Introduce la segunda fecha del rango: ")
            query_embedding = embedding_model.encode(question).tolist()
            relevant_news = chroma.searchByRangesDate(query_embedding=query_embedding, date=date, date2=date2,
                                                      top_k=data.get("retrieved_docs"))
            raw_context = (relevant_news.get("documents"))
            metadata = relevant_news.get("metadatas")
            context = []
            for news, meta in zip(raw_context[0], metadata[0]):
                context.append("La fecha del articulo es: " + str(meta['fecha']) + " " + str(news))
            answer = generate_response_with_ollama(question, str(context))
            # print(answer)
        elif choice == "6":
            print("Copiando los datos de Cassandra a ChromaDB...")
            migrate_cassandra_to_chroma()
            print("Hecho.")
        elif choice == "7":
            print("Evaluación de los datos obtenidos.")
            asyncio.run(test_evaluation())
        elif choice == "8":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción entre 1 y 8.")


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

    # context_recall = LLMContextRecall(llm=evaluator_llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    # await (context_recall.single_turn_ascore(sample))
    await context_precision.single_turn_ascore(sample2)


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        main_menu()
    finally:
        cassandra.close()
