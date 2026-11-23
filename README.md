# 🔍 RAG-Powered AIOps Log Anomaly Detector

An end-to-end **Retrieval-Augmented Generation (RAG)** system that detects anomalies in server and application logs, retrieves matching engineering runbooks from a vector database, and generates grounded root-cause analyses and remediation steps.

## Architecture

1. **Knowledge Base (`data/runbooks.json`)**: Stores known incident signatures, root causes, remediation steps, and normal traffic baselines.
2. **Embeddings & Vector Store (`src/retriever.py`)**: Embeds runbooks locally using `sentence-transformers/all-MiniLM-L6-v2` and indexes them in `ChromaDB`.
3. **RAG Diagnosis (`src/detector.py`)**: Retrieves the top-k most relevant runbooks for any incoming log and prompts `Gemini 2.5 Flash` to classify `NORMAL` vs. `ANOMALY`, assign severity, and recommend fixes.
4. **Interactive UI (`app.py`)**: Streamlit dashboard for testing sample logs or pasting custom logs, with an offline retrieval preview mode.

## Tech Stack

- **Language:** Python 3.10+
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB (persistent local storage)
- **LLM:** Google Gemini 2.5 Flash (`google-genai`)
- **Frontend:** Streamlit

## Quickstart

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd anomaly-detector-rag

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your free Gemini API key
# Create a .env file with: GEMINI_API_KEY=your_key_here

# 5. Launch the dashboard
streamlit run app.py