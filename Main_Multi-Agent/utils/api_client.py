import requests
from dotenv import load_dotenv
import os
import time

load_dotenv()

class APIClient:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.llm_endpoint = os.getenv("LLM_ENDPOINT")
        self.embed_endpoint = os.getenv("EMBED_ENDPOINT")
        self.rerank_endpoint = os.getenv("RERANK_ENDPOINT")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.retries = 3
        self.backoff_factor = 2

    def get_embedding(self, text):
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.embed_endpoint,
                    json={"input": text},
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                return response.json().get("embeddings", [])
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [400, 502]:
                    print(f"Embedding API error on attempt {attempt + 1}/{self.retries}: {e}")
                    if attempt < self.retries - 1:
                        time.sleep(self.backoff_factor ** attempt)
                        continue
                print(f"Embedding API error: {e}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"Embedding network error: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                return None
        return None

    def rerank(self, query, documents):
        if not documents:
            print("No documents to rerank")
            return documents
        try:
            response = requests.post(
                self.rerank_endpoint,
                json={"query": query, "documents": documents, "top_n": len(documents)},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("reranked", data.get("documents", documents))
        except requests.exceptions.HTTPError as e:
            print(f"Rerank API error: {e}")
            return documents
        except requests.exceptions.RequestException as e:
            print(f"Rerank network error: {e}")
            return documents

    def call_llm(self, prompt, max_tokens=500):
        try:
            response = requests.post(
                self.llm_endpoint,
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print(f"LLM API error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"LLM network error: {e}")
            return None