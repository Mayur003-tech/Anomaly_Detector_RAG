import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "runbooks.json"
CHROMA_DIR = BASE_DIR / ".chroma_db"


class RunbookRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name="incident_runbooks")

        if self.collection.count() == 0:
            self._index_runbooks()

    def _index_runbooks(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            runbooks = json.load(f)

        documents, metadatas, ids = [], [], []
        for rb in runbooks:
            category = rb.get("category", "General")
            text = (
                f"Title: {rb['title']}\n"
                f"Category: {category}\n"
                f"Symptoms: {rb['symptoms']}\n"
                f"Root Cause: {rb['root_cause']}\n"
                f"Remediation: {rb['remediation']}"
            )
            documents.append(text)
            metadatas.append({"id": rb["id"], "title": rb["title"], "category": category})
            ids.append(rb["id"])

        embeddings = self.embedder.encode(documents).tolist()
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        print(f"Indexed {len(ids)} runbooks into ChromaDB.")

    def retrieve(self, log_line: str, top_k: int = 2) -> list[dict]:
        """Find the top_k most similar runbooks for a given log line."""
        query_vector = self.embedder.encode([log_line]).tolist()
        results = self.collection.query(query_embeddings=query_vector, n_results=top_k)

        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = round(max(0.0, 1.0 - (distance / 2.0)) * 100, 1)
            matches.append({
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "category": results["metadatas"][0][i]["category"],
                "similarity_pct": similarity,
                "document": results["documents"][0][i],
            })
        return matches

    def add_runbook(self, title: str, category: str, symptoms: str, root_cause: str, remediation: str) -> str:
        """Add a new runbook dynamically to ChromaDB."""
        new_id = f"RB-{self.collection.count() + 1:03d}"
        text = (
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Symptoms: {symptoms}\n"
            f"Root Cause: {root_cause}\n"
            f"Remediation: {remediation}"
        )
        embedding = self.embedder.encode([text]).tolist()
        self.collection.add(
            ids=[new_id],
            documents=[text],
            embeddings=embedding,
            metadatas=[{"id": new_id, "title": title, "category": category}],
        )
        return new_id