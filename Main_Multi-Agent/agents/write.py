from utils.api_client import APIClient

class WriteAgent:
    def __init__(self):
        self.api_client = APIClient()

    def process(self, summary, critique):
        prompt = f"Based on this summary and critique, write a polished response:\nSummary: {summary}\nCritique: {critique}"
        return self.api_client.call_llm(prompt, max_tokens=300)