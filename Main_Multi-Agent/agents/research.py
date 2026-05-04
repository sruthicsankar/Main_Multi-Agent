from utils.retriever import Retriever

class ResearchAgent:
    def __init__(self):
        self.retriever = Retriever()

    def process(self, query, documents):
        try:
            self.retriever.index_documents(documents)
            return self.retriever.retrieve(query, top_k=5)
        except Exception as e:
            print(f"Research agent error: {e}")
            return []