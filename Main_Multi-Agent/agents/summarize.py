from utils.api_client import APIClient

class SummarizeAgent:
    def __init__(self):
        self.api_client = APIClient()

    def process(self, documents):
        if not documents:
            return "No documents provided for summarization"
        prompt = f"Summarize the following documents: {documents}"
        summary = self.api_client.call_llm(prompt)
        return summary if summary else "Summary could not be generated"