from datetime import timedelta, datetime

import chromadb
from chromadb.config import Settings


class ChromaDB:
    def __init__(self, collection_name):
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def insert_article(self, doc_id, metadata, content, embedding):
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding]
        )

    def search(self, query_embedding, top_k=3):
        """
        Busca en la colección utilizando un embedding de consulta.
        :param query_embedding: Vector de embedding de la consulta.
        :param top_k: Número de resultados a devolver.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def searchByRangesDate(self, query_embedding, date, date2, top_k=3):
        """
        Busca en la colección utilizando un embedding de consulta y un rango de fechas.
        :param date: Fecha de la que se querrá obtener la consulta del rango de fechas.
        :param date2: Fecha de la que se querrá obtener la consulta del rango de fechas.
        :param query_embedding: Vector de embedding de la consulta.
        :param top_k: Número de resultados a devolver.
        """
        try:
            date_ = datetime.strptime(date, '%d/%m/%Y')
            date_2 = datetime.strptime(date2, '%d/%m/%Y').timestamp()
        except ValueError:
            print("Introduzca bien el formato dd/mm/aaaa")
            return None
        date_ = date_.timestamp()

        if date_2 < date_:
            aux = date_
            date_ = date_2
            date_2 = aux
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "$and": [
                    {"fecha": {"$gte": date_}},
                    {"fecha": {"$lte": date_2}}
                ]

            }
        )

    def searchByDate(self, query_embedding, date, top_k=3):
        """
        Busca en la colección utilizando un embedding de consulta y una fecha.
        :param date: Fecha de la que se querrá obtener la consulta.
        :param query_embedding: Vector de embedding de la consulta.
        :param top_k: Número de resultados a devolver.
        """
        try:
            date_ = datetime.strptime(date, '%d/%m/%Y')
        except ValueError:
            print("Introduzca bien el formato dd/mm/aaaa")
            return None

        date_plus = (date_ + timedelta(days=2)).timestamp()
        date_minus = (date_ - timedelta(days=2)).timestamp()
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "$and": [
                    {"fecha": {"$gte": date_minus}},
                    {"fecha": {"$lte": date_plus}}
                ]
            }
        )

    def exists_by_title(self, titular_):
        """
        Verifica si un artículo con el título dado existe en la base de datos.
        :param titular_: Titular del artículo a buscar.
        :return: True si el artículo existe, False en caso contrario.
        """
        results = self.collection.query(
            query_texts=[titular_],
            n_results=1,
            where={"titular": titular_}
        )
        return results.get("documents", []) != [[]]
