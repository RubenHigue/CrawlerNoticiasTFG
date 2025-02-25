import ast
import asyncio
import re

import pandas as pd

import csv
import sys
import uuid
from datetime import datetime

from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

from ragas import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithoutReference, Faithfulness, context_recall, \
    context_precision, faithfulness
from ragas import evaluate
from ragas.evaluation import EvaluationDataset
from fuzzywuzzy import fuzz

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


# Funcion para cargar los datos de evaluación
def load_samples_from_csv(file_path):
    df = pd.read_csv(file_path)
    samples = []
    for _, row in df.iterrows():
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            reference=row["reference"],
            retrieved_contexts=ast.literal_eval(row["retrieved_contexts"]) if isinstance(row["retrieved_contexts"],
                                                                                         str) else row[
                "retrieved_contexts"]
        )
        samples.append(sample)
    return samples


# Funcion que ejecuta la evaluacion del proyecto
async def test_evaluation():
    file_path = "csv/evaluacion_tfg.csv"
    samples = load_samples_from_csv(file_path)

    run_config = RunConfig(max_retries=1)

    evaluator_llm = BaseLLMOllama(model_name=data.get("judge_model"), run_config=run_config)

    context_recall = LLMContextRecall(llm=evaluator_llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)

    '''
    OpenAI RAGAS
    dataset = EvaluationDataset(samples)
    results = evaluate(dataset, [context_recall, context_precision, faithfulness])
    print(results)
    '''

    for sample in samples:
        print("Evaluating sample:", sample.user_input)

        print("Context Recall Results: ")
        await context_recall.single_turn_ascore(sample)

        print("Context Precision Results: ")
        await context_precision.single_turn_ascore(sample)

        print("Faithfulness Results: ")
        await faithfulness.single_turn_ascore(sample)


# Función para consultar al LLM la precision y el recall del contexto
def query_llm_for_precision_recall(context, reference):
    promptPrecision = f"Contexto: {context} " + f"Referencia: {reference}" + data.get('prompt_eval_precision')
    promptRecall = f"Contexto: {context} " + f"Referencia: {reference}" + data.get('prompt_eval_recall')

    try:
        responsePrecision = ollama.chat(
            model=data.get("judge_model"),
            messages=[{'role': 'user', 'content': promptPrecision}]
        )

        responseRecall = ollama.chat(
            model=data.get("judge_model"),
            messages=[{'role': 'user', 'content': promptRecall}]
        )

        resultPrecision = responsePrecision.get('message')['content'].strip()
        #print(f'Precision: {resultPrecision}')

        resultRecall = responseRecall.get('message')['content'].strip()
        #print(f'Recall: {resultRecall}')

        precision = float(re.findall(r'\d+\.\d+', resultPrecision)[0])
        recall = float(re.findall(r'\d+\.\d+', resultRecall)[0])
        return precision, recall
    except Exception as e:
        print(f"Error al consultar LLM: {e}")
        return None, None


# Función de evaluación sin RAGAS
def evaluate_dataset_with_llm(csv_path):
    df = pd.read_csv(csv_path)
    precisions, recalls = [], []

    for _, row in df.iterrows():
        context = " ".join(eval(row["retrieved_contexts"])) if isinstance(row["retrieved_contexts"], str) else row[
            "retrieved_contexts"]
        reference = row["reference"]

        precision, recall = query_llm_for_precision_recall(context, reference)

        if precision is not None and recall is not None:
            precisions.append(precision)
            recalls.append(recall)

            print(f"Consulta: {row['user_input']}")
            print(f"Precisión: {precision:.2f}, Recall: {recall:.2f}")
            print("-" * 50)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    print(f"\nPrecisión promedio: {avg_precision:.2f}")
    print(f"Recall promedio: {avg_recall:.2f}")


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        if data.get("execution_mode") == "production":
            migrate_cassandra_to_chroma()
            app = QApplication(sys.argv)
            window = RAGDefensaApp(response_without_dates, response_with_dates, crawl_and_store,
                                   migrate_cassandra_to_chroma, test_evaluation, get_articles_from_db)
            window.show()
            sys.exit(app.exec())
        elif data.get("execution_mode") == "test":
            print("Evaluando base de datos...")
            #asyncio.run(test_evaluation())
            evaluate_dataset_with_llm(data.get("test_data"))
    finally:
        cassandra.close()
