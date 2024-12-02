import chromadb
from chromadb.config import Settings


class ChromaDB:
    def __init__(self, collection_name):
        self.client = chromadb.Client(Settings())
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def insert_article(self, doc_id, metadata, content, embedding):
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding]
        )

    def search(self, query_embedding, top_k=5):
        """
        Busca en la colección utilizando un embedding de consulta.
        :param query_embedding: Vector de embedding de la consulta.
        :param top_k: Número de resultados a devolver.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
