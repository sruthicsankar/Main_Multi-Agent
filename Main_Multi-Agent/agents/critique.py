from utils.api_client import APIClient

class CritiqueAgent:
    def __init__(self):
        self.api_client = APIClient()

    def process(self, summary):
        prompt = f"Evaluate the following summary for accuracy and completeness. Suggest improvements:\n{summary}"
        return self.api_client.call_llm(prompt, max_tokens=200)