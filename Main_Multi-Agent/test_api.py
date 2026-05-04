# test_api.py
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def test_api():
    headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}
    try:
        llm_response = requests.post(
            os.getenv("LLM_ENDPOINT"),
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test prompt"}], "max_tokens": 10},
            headers=headers,
            timeout=30
        )
        llm_response.raise_for_status()
        print("LLM Response:", llm_response.status_code, llm_response.json())
    except requests.exceptions.RequestException as e:
        print(f"LLM Error: {e}")
    try:
        embed_response = requests.post(
            os.getenv("EMBED_ENDPOINT"),
            json={"input": "Test text"},
            headers=headers,
            timeout=30
        )
        embed_response.raise_for_status()
        print("Embed Response:", embed_response.status_code, embed_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Embed Error: {e}")
    try:
        rerank_response = requests.post(
            os.getenv("RERANK_ENDPOINT"),
            json={"query": "Test query", "documents": ["Doc1", "Doc2"]},
            headers=headers,
            timeout=30
        )
        rerank_response.raise_for_status()
        print("Rerank Response:", rerank_response.status_code, rerank_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Rerank Error: {e}")

if __name__ == "__main__":
    test_api()