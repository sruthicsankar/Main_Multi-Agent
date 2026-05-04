## Testing
1. Activate virtual environment: `.\env\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `sample.pdf` exists in the project root.
4. Run tests: `pytest -v` (all 8 tests pass with mocked API responses).
5. Run end-to-end workflow: `python main.py` (uses local `sentence-transformers`, `cross-encoder`, and `distilgpt2` to bypass API errors).
6. Test multiple queries by editing `query` in `main.py`.
7. View results for queries like "What is the main topic?" or "Which development of ERP systems in the 1960s??".

Note: Please add .env file