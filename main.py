import ast
import asyncio
import re

import pandas as pd
from flask import Flask, render_template, request, jsonify
from langchain.chains.natbot.crawler import Crawler

import csv
import sys
import uuid

from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

from ragas.dataset_schema import SingleTurnSample
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import spacy

import ollama

from Crawler import crawl_website
from BaseDeDatos.Cassandra import Cassandra
from BaseDeDatos.ChromaDB import ChromaDB
from sentence_transformers import SentenceTransformer

from Interface.RAGDefensaApp import RAGDefensaApp
import yaml

load_dotenv()

nlp = spacy.load("es_core_news_sm")

# Carga de los datos de config
with open("config_data.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
    urls = data["news_url"].split(", ")

# Inicializar bases de datos
cassandra = Cassandra(hosts=[data.get("cassandra_host")], keyspace=data.get("keyspace"))
chroma = ChromaDB(collection_name="noticias_articulos")

# Inicializar el modelo para embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Contextos

last_context = None
last_context_date = None

# Fecha actualizacion
last_crawl_date = None


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
    # print(response.get('message', "No response received"))
    # print(response2.get('message', "No response received."))
    answer = response.get('message', "No response received.")
    answer = answer.get("content")
    return answer


from sentence_transformers import CrossEncoder

# Inicializa el modelo CrossEncoder (puedes hacerlo en global)
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')


# Funcion para reordenar los documentos obtenidos de ChromaDB
def rerank_documents(question, retrieved_docs, top_n=5):
    raw_context = retrieved_docs.get("documents")
    pairs = [(question, doc) for doc in raw_context[0]]

    scores = cross_encoder.predict(pairs)

    docs_with_scores = list(zip(raw_context[0], scores))

    reranked = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in reranked[:top_n]]

    return reranked_docs


# Funcion para consultas sin fechas
def response_without_dates(question):
    query_embedding = embedding_model.encode(question).tolist()
    relevant_news = chroma.search(query_embedding=query_embedding, top_k=data.get("retrieved_docs"))
    context = process_relevant_news_rerank(relevant_news, question, data.get("re_ranked_docs"))
    answer = generate_response_with_ollama(question, str(context))
    return answer, context


# Funcion para consultas con fechas
def response_with_dates(question, date, date2):
    if date != date2:
        query_embedding = embedding_model.encode(question).tolist()
        relevant_news = chroma.searchByRangesDate(query_embedding=query_embedding, date=date, date2=date2,
                                                  top_k=data.get("retrieved_docs"))
        context = process_relevant_news_rerank(relevant_news, question, data.get("re_ranked_docs"))
        answer = generate_response_with_ollama(question, str(context))
        return answer, context
    else:
        query_embedding = embedding_model.encode(question).tolist()
        relevant_news = chroma.searchByDate(query_embedding=query_embedding, date=date,
                                            top_k=data.get("retrieved_docs"))
        context = process_relevant_news_rerank(relevant_news, question, data.get("re_ranked_docs"))
        answer = generate_response_with_ollama(question, str(context))
        return answer, context


# Funcion para procesar el contexto y sus metadatos para el modelo
def process_relevant_news(relevant_news):
    raw_context = relevant_news.get("documents")
    metadata = relevant_news.get("metadatas")
    context = []
    for news, meta in zip(raw_context[0], metadata[0]):
        context.append(
            "La fecha del artículo es: " + datetime.utcfromtimestamp(meta['fecha']).strftime('%d/%m/%Y') + " " + str(
                news))  + " Fuente: " + meta['fuente']
    return context


# Funcion para procesar el contexto y sus metadatos para el modelo con reranking
def process_relevant_news_rerank(relevant_news, question, top_n=5):
    raw_context = relevant_news.get("documents")[0]
    metadata = relevant_news.get("metadatas")[0]

    pairs = [(question, doc) for doc in raw_context]
    scores = cross_encoder.predict(pairs)

    combined = list(zip(raw_context, metadata, scores))
    sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)

    top_docs = sorted_combined[:top_n]

    context = []
    for news, meta, _ in top_docs:
        context.append(
            "La fecha del artículo es: " +
            datetime.utcfromtimestamp(meta['fecha']).strftime('%d/%m/%Y') +
            " " + str(news) + " Fuente: " + meta['fuente']
        )

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


# Función para segmentar los articulos en párrafos
def segmentate_spacy(texto):
    doc = nlp(texto)
    parrafos = [sent.text for sent in doc.sents]
    return parrafos


# Función para pasar los articulos de Cassandra a Chroma por párrafos
def migrate_cassandra_to_chroma():
    all_articles = cassandra.get_all_entities(
        table=data.get("table_name")
    )

    print(f"Se encontraron {len(all_articles)} artículos en Cassandra para migrar a Chroma.")

    for article in all_articles:
        if not chroma.exists_by_title(article["titular"]):
            print(f"Migrando artículo con ID: {article['id']} - Titular: {article['titular']}")

            paragraphs = segmentate_spacy(article["noticia"])

            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():  # Evitar insertar párrafos vacíos
                    embedding = embedding_model.encode(paragraph)

                    chroma.insert_article(
                        doc_id=f"{article['id']}_{i}",  # Identificador único por párrafo
                        metadata={
                            "fuente": article["fuente"],
                            "fecha": datetime.strptime(article["fecha"], "%d/%m/%Y").timestamp() if article[
                                                                                                        "fecha"] != "Fecha no encontrada" else datetime.now().timestamp(),
                            "titular": article["titular"],
                            "url": article["url"]
                        },
                        content=paragraph,
                        embedding=embedding
                    )
        else:
            print(f"El artículo con titular: {article['titular']} ya está en la base de datos ChromaDB.")

    print("Migración completa. Todos los artículos han sido insertados en Chroma por párrafos.")


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


# Función para consultar al LLM la precision y el recall del contexto
def query_llm_for_precision_recall(context, reference, judge_model):
    promptPrecision = f"Contexto: {context} " + f"Referencia: {reference}" + data.get('prompt_eval_precision')
    promptRecall = f"Contexto: {context} " + f"Referencia: {reference}" + data.get('prompt_eval_recall')

    try:
        responsePrecision = ollama.chat(
            model=judge_model,
            messages=[{'role': 'user', 'content': promptPrecision}]
        )

        responseRecall = ollama.chat(
            model=judge_model,
            messages=[{'role': 'user', 'content': promptRecall}]
        )

        resultPrecision = responsePrecision.get('message')['content'].strip()
        # print(f'Precision: {resultPrecision}')

        resultRecall = responseRecall.get('message')['content'].strip()
        # print(f'Recall: {resultRecall}')

        precision = float(re.findall(r'\d+\.\d+', resultPrecision)[0])
        recall = float(re.findall(r'\d+\.\d+', resultRecall)[0])
        return precision, recall
    except Exception as e:
        print(f"Error al consultar LLM: {e}")
        return None, None


# Función para consultar al LLM la FactualCorrectness y la similarity del contexto
def query_llm_for_factual_correctness_similarity(answer, reference, judge_model):
    promptFactualCorrectness = f"Respuesta: {answer} " + f"Referencia: {reference}" + data.get(
        'prompt_eval_factual_correctness')
    promptSimilarity = f"Respuesta: {answer} " + f"Referencia: {reference}" + data.get('prompt_eval_similarity')

    try:
        responseFactualCorrectness = ollama.chat(
            model=judge_model,
            messages=[{'role': 'user', 'content': promptFactualCorrectness}]
        )

        resultFactualCorrectness = responseFactualCorrectness.get('message')['content'].strip()
        # print(f'FactualCorrectness: {resultFactualCorrectness}')

        answer_embedding = embedding_model.encode(answer).tolist()
        reference_embedding = embedding_model.encode(reference).tolist()

        if reference_embedding is not None and answer_embedding is not None:
            similarity = cosine_similarity([reference_embedding], [answer_embedding])[0][0]
        else:
            similarity = None

        factual_correctness = float(re.findall(r'\d+\.\d+', resultFactualCorrectness)[0])
        return factual_correctness, similarity
    except Exception as e:
        print(f"Error al consultar LLM: {e}")
        return None, None


# Función para la evaluación de la fidelidad de las respuestas
def check_faithfulness_with_ollama(answer, context, judge_model):
    prompt = f"Contexto: {context} " + f"Respuesta: {answer}" + data.get('prompt_eval_faith')

    try:
        response = ollama.chat(
            model=judge_model,
            messages=[{'role': 'user', 'content': prompt}]
        )

        result = response.get('message')['content'].strip()

        return result
    except Exception as e:
        print(f"Error al consultar Ollama: {e}")
        return None


# Funcion para calcular el Recall en función de las referencias de los textos.
def compute_context_recall(reference_context, retrieved_contexts):
    if not reference_context or not retrieved_contexts:
        return 0.0

    found_count = sum(1 for fragment in reference_context if fragment.lower() in retrieved_contexts.lower())

    recall = found_count / len(reference_context)
    return recall


# Cargar modelos juez desde config
def load_judge_models():
    return data["judge_models"].split(", ")


# Función de evaluación sin RAGAS
def evaluate_dataset_with_llm(csv_path):
    df = pd.read_csv(csv_path)
    judge_models = load_judge_models()

    for index, row in df.iterrows():
        context = " ".join(eval(row["retrieved_contexts"])) if isinstance(row["retrieved_contexts"], str) else row[
            "retrieved_contexts"]
        reference = row["reference"]
        response = row["response"]
        reference_response = row["reference"]
        reference_context = ast.literal_eval(row["reference_context"]) if isinstance(row["reference_context"], str) else \
            row["reference_context"]

        print(f"Consulta: {row['user_input']}")

        for judge in judge_models:
            precision, recall = query_llm_for_precision_recall(context, reference, judge_model=judge)
            factualcorrectness, similarity = query_llm_for_factual_correctness_similarity(response, reference_response,
                                                                                          judge_model=judge)
            recallreal = compute_context_recall(reference_context, context)
            faith = check_faithfulness_with_ollama(response, context, judge_model=judge)

            if precision is not None and recall is not None and factualcorrectness is not None and similarity is not None:
                df.at[index, f"precision_{judge}"] = precision
                df.at[index, f"recall_{judge}"] = recall
                df.at[index, f"factual_correctness_{judge}"] = factualcorrectness
                df.at[index, f"faithfulness_{judge}"] = faith

                df.at[index, "real_recall"] = recallreal
                df.at[index, f"similarity"] = similarity

                print(f"Modelo juez: {judge}")
                print(f"Precisión: {precision:.2f}, Recall: {recall:.2f}")
                print(f"Factual Correctness: {factualcorrectness:.2f}, Similarity: {similarity:.2f}")

            print(f"Real Recall: {recallreal:.2f}, Faithfulness: {faith}")
            print("-" * 50)
            df.to_csv(csv_path, index=False)

    # Calcular promedios por modelo juez
    for judge in judge_models:
        precisions = df[f"precision_{judge}"].dropna()
        recalls = df[f"recall_{judge}"].dropna()
        factuals = df[f"factual_correctness_{judge}"].dropna()
        faiths = df[f"faithfulness_{judge}"].dropna()

        avg_precision = precisions.mean() if not precisions.empty else 0
        avg_recall = recalls.mean() if not recalls.empty else 0
        avg_fact = factuals.mean() if not factuals.empty else 0

        print(f"\n[Métricas promedio para {judge}]")
        print(f"Precisión: {avg_precision:.2f}")
        print(f"Recall: {avg_recall:.2f}")
        print(f"Factual Correctness: {avg_fact:.2f}")

    # Real recall y faithfulness promedio generales
    realrecalls = df["real_recall"].dropna()
    similarity = df["similarity"].dropna()

    avg_realrecall = realrecalls.mean() if not realrecalls.empty else 0
    avg_sim = similarity.mean() if not similarity.empty else 0

    print(f"\nRecall real promedio: {avg_realrecall:.2f}")
    print(f"Similarity promedio: {avg_sim:.2f}")

def obtain_last_crawl_date():
    rows = cassandra.get_last_crawl_date('noticias_tabulares')

    fechas = []
    global last_crawl_date

    for row in rows:
        try:
            fecha_obj = datetime.strptime(row['fecha'], "%d/%m/%Y")
            fechas.append(fecha_obj)
        except:
            pass

    if fechas:
        return max(fechas).strftime("%d/%m/%Y")
    else:
        return "Error en la información"


''''APP FLASK'''
app = Flask(__name__)


@app.route('/')
def index():
    global last_crawl_date
    last_crawl_date = obtain_last_crawl_date()
    return render_template('query_tab.html', last_crawl_date=last_crawl_date)


@app.route('/query_tab', methods=['GET', 'POST'])
def query_tab():
    answer = None
    context = None

    if request.method == 'POST':
        question = request.form.get('query')
        if question:
            answer, context = response_without_dates(question)
        else:
            answer = "Por favor, introduce una consulta."
            context = []

    return render_template('query_tab.html', answer=answer, context=context, last_crawl_date=last_crawl_date)


@app.route('/query_with_date_tab', methods=['GET', 'POST'])
def query_with_date_tab():
    answer = None
    context = None

    if request.method == 'POST':
        question = request.form.get('query_date')
        date1 = request.form.get('date1')
        date2 = request.form.get('date2')

        print(date1)
        print(date2)

        try:
            date1 = datetime.strptime(date1, "%Y-%m-%d").strftime("%d/%m/%Y")
            date2 = datetime.strptime(date2, "%Y-%m-%d").strftime("%d/%m/%Y")
        except ValueError:
            return render_template('query_with_date_tab.html', answer="Formato de fecha incorrecto", context=[], last_crawl_date=last_crawl_date)

        print(f"Consulta: {question}, Fecha Inicio: {date1}, Fecha Fin: {date2}")  # DEBUG

        if question and date1 and date2:
            try:
                answer, context = response_with_dates(question, date1, date2)
                print(f"Respuesta: {answer}, Contexto: {context}")  # DEBUG
            except Exception as e:
                print(f"Error en response_with_dates: {e}")  # DEBUG
                answer = "Error al procesar la consulta."
                context = []
        else:
            answer = "Por favor, completa la consulta y selecciona las fechas."

    return render_template('query_with_date_tab.html', answer=answer, context=context, last_crawl_date=last_crawl_date)


@app.route('/documents_tab')
def documents_tab():
    page = request.args.get('page', 1, type=int)
    articles_per_page = 10  # Artículos por página
    all_articles = get_articles_from_db(data.get("table_name"))

    total_pages = (len(all_articles) + articles_per_page - 1) // articles_per_page
    start = (page - 1) * articles_per_page
    end = start + articles_per_page
    articles = all_articles[start:end]

    return render_template("documents_tab.html", articles=articles, page=page, total_pages=total_pages, last_crawl_date=last_crawl_date)


@app.route('/execute_query', methods=['POST'])
def execute_query():
    global last_context
    question = request.form['query']
    if not question:
        return render_template('query_tab.html', query=question, answer="Por favor, introduce una consulta.")

    answer, context = response_without_dates(question)

    last_context = context

    return render_template('query_tab.html', query=question, answer=answer, last_crawl_date=last_crawl_date)


@app.route('/execute_query_with_date', methods=['POST'])
def execute_query_with_date():
    global last_context_date
    question = request.form['query_date']
    date1 = request.form['date1']
    date2 = request.form['date2']

    try:
        date1 = datetime.strptime(date1, "%Y-%m-%d").strftime("%d/%m/%Y")
        date2 = datetime.strptime(date2, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return render_template('query_with_date_tab.html', answer="Formato de fecha incorrecto", context=[])

    print(f"Consulta: {question}, Fecha Inicio: {date1}, Fecha Fin: {date2}")  # DEBUG

    if question and date1 and date2:
        try:
            answer, context = response_with_dates(question, date1, date2)
            print(f"Respuesta: {answer}, Contexto: {context}")  # DEBUG
            last_context_date = context
        except Exception as e:
            print(f"Error en response_with_dates: {e}")  # DEBUG
            answer = "Error al procesar la consulta."
            context = []
    else:
        answer = "Por favor, completa la consulta y selecciona las fechas."

    return render_template('query_with_date_tab.html', answer=answer, context=context, last_crawl_date=last_crawl_date)


@app.route('/run_scraping', methods=['POST'])
def run_scraping():
    crawl_and_store(urls=urls)
    global last_crawl_date
    last_crawl_date = obtain_last_crawl_date()
    return "Scraping ejecutado correctamente."


@app.route('/migrate_data', methods=['POST'])
def migrate_data():
    migrate_cassandra_to_chroma()
    return "Datos migrados a ChromaDB."


@app.route('/show_context', methods=['POST'])
def show_context():
    if not last_context:
        return jsonify({"error": "No hay contexto disponible. Ejecuta una consulta primero."}), 400
    return jsonify({"title": "Contexto Usado (Sin Fecha)", "context": "\n\n---\n\n".join(last_context)})


@app.route('/show_context_date', methods=['POST'])
def show_context_date():
    if not last_context_date:
        return jsonify({"error": "No hay contexto disponible. Ejecuta una consulta con fecha primero."}), 400
    return jsonify({"title": "Contexto Usado (Con Fecha)", "context": "\n\n---\n\n".join(last_context_date)})


# Ejecutar el proceso
if __name__ == "__main__":
    try:
        if data.get("execution_mode") == "production":
            '''
            app = QApplication(sys.argv)
            window = RAGDefensaApp(response_without_dates, response_with_dates, crawl_and_store,
                                   migrate_cassandra_to_chroma, get_articles_from_db)
            window.show()
            sys.exit(app.exec())
            '''
            app.run(debug=True)
        elif data.get("execution_mode") == "test":
            print("Evaluando base de datos...")
            # asyncio.run(test_evaluation())
            evaluate_dataset_with_llm(data.get("test_data"))
    finally:
        cassandra.close()
