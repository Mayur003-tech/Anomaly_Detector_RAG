import json
import os
import re
import time
import requests
from dotenv import load_dotenv
from google import genai
from src.retriever import RunbookRetriever

load_dotenv()


class AnomalyDetector:
    def __init__(self):
        self.retriever = RunbookRetriever()

    def _call_groq(self, prompt: str, groq_key: str) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_json_response(self, raw_text: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "status": "ANOMALY",
                "severity": "Medium",
                "summary": "AI analysis completed.",
                "root_cause": raw_text,
                "impact": "Review the retrieved runbook for system impact.",
                "remediation_steps": ["Inspect the matched runbook below and verify service health."],
                "prevention_tip": "Monitor this service metric with automated alerts.",
            }

    def analyze(self, log_line: str, top_k: int = 2, api_key: str | None = None) -> dict:
        matches = self.retriever.retrieve(log_line, top_k=top_k)
        context = "\n\n---\n\n".join(m["document"] for m in matches)

        prompt = f"""You are an expert Site Reliability Engineer (SRE) mentoring a junior engineer.
Analyze the incoming log using ONLY the retrieved runbook context below.
Return ONLY a valid JSON object (no markdown fences) with these exact keys:
{{
  "status": "NORMAL" or "ANOMALY",
  "severity": "Low" or "Medium" or "High" or "Critical",
  "summary": "One clear sentence summarizing what happened.",
  "root_cause": "2-3 beginner-friendly sentences explaining why this happened based on the runbook.",
  "impact": "1-2 sentences explaining what breaks for users or services if unfixed (or 'None' if normal).",
  "remediation_steps": ["Step 1...", "Step 2...", "Step 3..."],
  "prevention_tip": "One practical tip to prevent this issue in the future."
}}

Incoming Log:
{log_line}

Retrieved Runbooks:
{context}
"""

        groq_key = api_key if (api_key and api_key.startswith("gsk_")) else os.getenv("GROQ_API_KEY")
        gemini_key = api_key if (api_key and not api_key.startswith("gsk_")) else os.getenv("GEMINI_API_KEY")

        start = time.time()
        if groq_key:
            try:
                raw = self._call_groq(prompt, groq_key)
                latency = round(time.time() - start, 2)
                return {
                    "log": log_line,
                    "matches": matches,
                    "structured": self._parse_json_response(raw),
                    "prompt": prompt,
                    "latency_s": latency,
                    "mode": f"⚡ Groq (openai/gpt-oss-20b) — {latency}s",
                }
            except Exception as e:
                fallback_note = f"Groq error: {e}"
        elif gemini_key:
            try:
                client = genai.Client(api_key=gemini_key)
                resp = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
                latency = round(time.time() - start, 2)
                return {
                    "log": log_line,
                    "matches": matches,
                    "structured": self._parse_json_response(resp.text),
                    "prompt": prompt,
                    "latency_s": latency,
                    "mode": f"Gemini 3.8 Flash — {latency}s",
                }
            except Exception as e:
                fallback_note = f"Gemini error: {e}"
        else:
            fallback_note = "Add GROQ_API_KEY or GEMINI_API_KEY to enable live AI analysis."

        top = matches[0]
        is_normal = "Normal" in top["title"]
        return {
            "log": log_line,
            "matches": matches,
            "structured": {
                "status": "NORMAL" if is_normal else "ANOMALY",
                "severity": "Low" if is_normal else "High",
                "summary": f"Matched {top['id']} — {top['title']} ({top['similarity_pct']}% match).",
                "root_cause": fallback_note,
                "impact": "No impact." if is_normal else "Potential service degradation.",
                "remediation_steps": ["Review the retrieved runbook context below."],
                "prevention_tip": "Configure live LLM API key for detailed guidance.",
            },
            "prompt": prompt,
            "latency_s": 0.0,
            "mode": "Fallback Retrieval Mode",
        }

    def ask_followup(self, log_line: str, matches: list[dict], question: str, api_key: str | None = None) -> str:
        """Interactive follow-up Q&A grounded in the same log and retrieved runbooks."""
        groq_key = api_key if (api_key and api_key.startswith("gsk_")) else os.getenv("GROQ_API_KEY")
        if not groq_key:
            return "Add your `GROQ_API_KEY` in the sidebar or `.env` to ask follow-up questions."

        context = "\n\n---\n\n".join(m["document"] for m in matches)
        prompt = f"""You are a helpful SRE mentor. Answer the user's follow-up question clearly and concisely using the log and runbook context below.

Log: {log_line}
Context:
{context}

Question: {question}
"""
        return self._call_groq(prompt, groq_key)

    def summarize_batch(self, anomaly_rows: list[str], api_key: str | None = None) -> str:
        groq_key = api_key if (api_key and api_key.startswith("gsk_")) else os.getenv("GROQ_API_KEY")
        if not groq_key:
            return "Add your `GROQ_API_KEY` in the sidebar or `.env` to generate an AI incident summary."

        anomalies_text = "\n".join(f"- {row}" for row in anomaly_rows)
        prompt = f"""You are a Lead SRE reviewing a batch log scan.
Summarize these flagged anomalies in 3-4 concise bullet points:
1. Main incident themes
2. Highest-priority issue to fix first
3. Recommended immediate action

Flagged Anomalies:
{anomalies_text}
"""
        return self._call_groq(prompt, groq_key)