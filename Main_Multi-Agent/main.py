from utils.document_processor import DocumentProcessor
from utils.retriever import Retriever
from agents.research import ResearchAgent
from agents.summarize import SummarizeAgent
from agents.critique import CritiqueAgent
from agents.write import WriteAgent
from langgraph.graph import StateGraph, END
from typing import Dict, Any

# Define state for the workflow
class WorkflowState(Dict[str, Any]):
    query: str
    documents: list
    research_output: list
    summary: str
    critique: str
    final_output: str

def main():
    processor = DocumentProcessor()
    retriever = Retriever()
    research_agent = ResearchAgent()
    summarize_agent = SummarizeAgent()
    critique_agent = CritiqueAgent()
    write_agent = WriteAgent()

    # Define LangGraph workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("research", lambda state: {"research_output": research_agent.process(state["query"], state["documents"])})
    workflow.add_node("summarize", lambda state: {"summary": summarize_agent.process(state["research_output"])})
    workflow.add_node("critique", lambda state: {"critique": critique_agent.process(state["summary"])})
    workflow.add_node("write", lambda state: {"final_output": write_agent.process(state["summary"], state["critique"])})
    workflow.add_edge("research", "summarize")
    workflow.add_edge("summarize", "critique")
    workflow.add_edge("critique", "write")
    workflow.add_edge("write", END)
    workflow.set_entry_point("research")
    app = workflow.compile()

    # Load and process document
    try:
        text = processor.load_pdf("sample.pdf")
        chunks = processor.chunk_text(text)
    except FileNotFoundError:
        print("Error: sample.pdf not found")
        return

    # Index documents
    try:
        retriever.index_documents(chunks)
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return

    # Run workflow
    query = "Which development of ERP systems in the 1960s?"
    try:
        result = app.invoke({"query": query, "documents": chunks})
        print(f"The main topic of the document is: {result['final_output']}")
    except Exception as e:
        print(f"Error running workflow: {e}")

if __name__ == "__main__":
    main()