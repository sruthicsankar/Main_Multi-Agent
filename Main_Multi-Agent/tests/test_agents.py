import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import requests
from utils.document_processor import DocumentProcessor
from utils.retriever import Retriever
from utils.api_client import APIClient
from agents.research import ResearchAgent
from agents.summarize import SummarizeAgent
from agents.critique import CritiqueAgent
from agents.write import WriteAgent
from chromadb import Client

@pytest.fixture
def sample_text():
    return "This is a test document about artificial intelligence and machine learning."

@pytest.fixture
def processor():
    return DocumentProcessor()

@pytest.fixture
def retriever():
    client = Client()
    try:
        client.delete_collection("documents")  # Try to reset collection
    except:
        pass  # Ignore if collection doesn't exist
    return Retriever()

@pytest.fixture
def api_client(monkeypatch):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
        def json(self):
            return self.json_data
        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.RequestException("Mock API error")

    def mock_post(*args, **kwargs):
        if "embed" in args[0]:
            return MockResponse({"embeddings": [0.1, 0.2, 0.3]}, 200)
        elif "reranker" in args[0]:
            return MockResponse({"reranked": kwargs["json"]["documents"]}, 200)
        elif "completions" in args[0]:
            return MockResponse({"choices": [{"text": "Mock response"}]}, 200)
        return MockResponse({}, 404)

    monkeypatch.setattr("requests.post", mock_post)
    return APIClient()

def test_document_processor(processor, sample_text):
    chunks = processor.chunk_text(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_document_processor_pdf(processor):
    try:
        text = processor.load_pdf("sample.pdf")
        assert isinstance(text, str)
        assert len(text) > 0
    except FileNotFoundError:
        pytest.skip("sample.pdf not found")

def test_retriever_index_and_retrieve(retriever, sample_text, api_client):
    retriever.api_client = api_client
    chunks = [sample_text]
    retriever.index_documents(chunks)
    results = retriever.retrieve("test query", top_k=1)
    assert len(results) == 1
    assert results[0] == sample_text

def test_research_agent(retriever, sample_text, api_client):
    retriever.api_client = api_client
    agent = ResearchAgent()
    agent.retriever = retriever
    chunks = [sample_text]
    results = agent.process("test query", chunks)
    assert len(results) > 0

def test_summarize_agent(api_client, sample_text):
    agent = SummarizeAgent()
    agent.api_client = api_client
    summary = agent.process([sample_text])
    assert summary == "Mock response"

def test_critique_agent(api_client):
    agent = CritiqueAgent()
    agent.api_client = api_client
    critique = agent.process("Test summary")
    assert critique == "Mock response"

def test_write_agent(api_client):
    agent = WriteAgent()
    agent.api_client = api_client
    response = agent.process("Test summary", "Test critique")
    assert response == "Mock response"

def test_api_client_error_handling(monkeypatch):
    class MockResponse:
        def __init__(self):
            self.status_code = 500
        def raise_for_status(self):
            raise requests.RequestException("API error")

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)
    client = APIClient()
    assert client.get_embedding("test") is None
    assert client.rerank("test", ["doc1"]) == ["doc1"]
    assert client.call_llm("test") is None