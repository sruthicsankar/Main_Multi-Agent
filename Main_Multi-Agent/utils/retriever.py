from chromadb import Client
from rank_bm25 import BM25Okapi
from utils.api_client import APIClient
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.client = Client()
        self.collection = self.client.get_or_create_collection("documents")
        self.api_client = APIClient()
        self.bm25 = None
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Failed to load local embedding model: {e}")
            self.local_model = None

    def index_documents(self, chunks):
        if not chunks:
            print("No chunks to index")
            return
        embeddings = []
        for chunk in chunks:
            embedding = self.api_client.get_embedding(chunk)
            if embedding is None or not embedding:
                if self.local_model is None:
                    print(f"Skipping chunk due to embedding failure: {chunk[:50]}...")
                    continue
                print(f"Using local embedding model for chunk: {chunk[:50]}...")
                embedding = self.local_model.encode(chunk).tolist()
            embeddings.append(embedding)
        if not embeddings:
            print("No valid embeddings generated")
            return
        self.collection.add(
            documents=chunks[:len(embeddings)],
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(embeddings))]
        )
        tokenized_chunks = [chunk.split() for chunk in chunks[:len(embeddings)]]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=5):
        query_embedding = self.api_client.get_embedding(query)
        if query_embedding is None or not query_embedding:
            if self.local_model is None:
                print("No valid query embedding; returning empty results")
                return []
            print("Using local embedding model for query")
            query_embedding = self.local_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        semantic_docs = results["documents"][0] if results["documents"] else []
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else [0] * len(semantic_docs)
        bm25_top_k = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_docs = [self.collection.get(ids=[f"doc_{i}"])["documents"][0] for i in bm25_top_k if i < len(semantic_docs)]
        combined_docs = list(set(semantic_docs + bm25_docs))
        reranked_docs = self.api_client.rerank(query, combined_docs)
        return reranked_docs[:top_k]