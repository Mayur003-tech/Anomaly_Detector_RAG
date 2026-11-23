import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.detector import AnomalyDetector

load_dotenv()

st.set_page_config(page_title="AIOps Log Anomaly Detector", page_icon="🔍", layout="wide")

st.title("🔍 RAG-Powered Log Anomaly Detector")
st.caption("Detects server & application anomalies and explains root causes using retrieved runbooks.")


@st.cache_resource
def get_detector() -> AnomalyDetector:
    return AnomalyDetector()


detector = get_detector()

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "LLM API Key (Groq or Gemini)",
        value=os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Paste a gsk_... Groq key or Gemini key. Leave blank for offline mode.",
    )
    top_k = st.slider("Retrieved Runbooks (Top-K)", min_value=1, max_value=4, value=2)
    st.markdown("---")
    st.markdown("**Pipeline:**")
    st.markdown("1. Embed log (`all-MiniLM-L6-v2`)\n2. Retrieve runbooks (`ChromaDB`)\n3. Diagnose (`Groq / Gemini`)")

sample_path = Path(__file__).parent / "data" / "sample_logs.csv"
default_df = pd.read_csv(sample_path)

tab1, tab2, tab3 = st.tabs(["🔎 Single Log Analyzer", "📊 Batch CSV Scanner", "📚 Add Runbook"])

# --- TAB 1: Single Log Analyzer ---
with tab1:
    col1, col2 = st.columns([1, 1.35])

    with col1:
        st.subheader("1. Choose or Paste a Log")
        mode = st.radio("Input:", ["Pick from sample_logs.csv", "Write custom log"], horizontal=True)

        if mode == "Pick from sample_logs.csv":
            options = [f"[{r['level']}] {r['service']}: {r['message']}" for _, r in default_df.iterrows()]
            selected_log = st.selectbox("Select a log entry:", options)
        else:
            selected_log = st.text_area(
                "Paste a server log:",
                value="[ERROR] db-proxy: FATAL: remaining connection slots are reserved",
                height=110,
            )

        if st.button("⚡ Analyze Log", type="primary", use_container_width=True):
            if selected_log.strip():
                with st.spinner("Retrieving runbooks & analyzing with Groq..."):
                    st.session_state["last_result"] = detector.analyze(
                        selected_log, top_k=top_k, api_key=api_key or None
                    )
                    st.session_state.pop("followup_answer", None)

        with st.expander("📄 Preview sample_logs.csv"):
            st.dataframe(default_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("2. RAG Diagnosis")
        result = st.session_state.get("last_result")

        if result:
            data = result["structured"]
            is_normal = data.get("status", "").upper() == "NORMAL"

            # Visual Status Banner
            if is_normal:
                st.success(f"✅ **NORMAL** — {data.get('summary', '')}")
            else:
                st.error(f"🚨 **ANOMALY DETECTED** — {data.get('summary', '')}")

            # KPI Metrics Row
            k1, k2, k3 = st.columns(3)
            k1.metric("Status", data.get("status", "ANOMALY"))
            k2.metric("Severity", data.get("severity", "Medium"))
            k3.metric("Groq Latency", f"{result['latency_s']}s" if result["latency_s"] else "Offline")

            # Rich Explanations
            with st.container(border=True):
                st.markdown("**🧠 Root Cause (Why This Happened)**")
                st.write(data.get("root_cause", ""))
                st.markdown("**💥 System Impact**")
                st.write(data.get("impact", ""))

            # Interactive Remediation Checklist
            with st.container(border=True):
                st.markdown("**🛠️ Interactive Remediation Checklist**")
                for idx, step in enumerate(data.get("remediation_steps", [])):
                    st.checkbox(step, key=f"fix_step_{idx}_{hash(result['log'])}")
                st.info(f"💡 **Prevention Tip:** {data.get('prevention_tip', '')}")

            # Interactive Follow-up Q&A with Groq
            st.markdown("**💬 Ask Groq About This Incident**")
            q1, q2 = st.columns(2)
            if q1.button("Explain like I'm a beginner", use_container_width=True):
                with st.spinner("Asking Groq..."):
                    st.session_state["followup_answer"] = detector.ask_followup(
                        result["log"], result["matches"], "Explain this error simply for a beginner.", api_key or None
                    )
            if q2.button("Give me terminal debug commands", use_container_width=True):
                with st.spinner("Asking Groq..."):
                    st.session_state["followup_answer"] = detector.ask_followup(
                        result["log"], result["matches"], "Give 2-3 exact terminal commands to debug this issue.", api_key or None
                    )

            custom_q = st.text_input("Or ask your own follow-up question:", placeholder="e.g., How do I prevent this in Kubernetes?")
            if custom_q:
                with st.spinner("Asking Groq..."):
                    st.session_state["followup_answer"] = detector.ask_followup(
                        result["log"], result["matches"], custom_q, api_key or None
                    )

            if "followup_answer" in st.session_state:
                st.info(st.session_state["followup_answer"])

            # Retrieved Context + Prompt Transparency
            st.subheader("3. Retrieved Knowledge & Prompt")
            for match in result["matches"]:
                with st.expander(f"📘 {match['id']} — {match['title']} ({match['similarity_pct']}% match)"):
                    st.text(match["document"])
            with st.expander("🔍 Inspect Augmented Prompt Sent to Groq"):
                st.code(result["prompt"], language="markdown")
        else:
            st.info("Pick a log on the left and click **⚡ Analyze Log** to see the full interactive diagnosis.")

# --- TAB 2: Batch CSV Scanner ---
with tab2:
    st.subheader("Scan an Entire Log File at Once")
    uploaded_csv = st.file_uploader("Upload a CSV with a 'message' column (or use built-in sample_logs.csv)", type=["csv"])
    scan_df = pd.read_csv(uploaded_csv) if uploaded_csv else default_df.copy()

    if st.button("Run Batch Scan", type="primary"):
        if "message" not in scan_df.columns:
            st.error("Your CSV must have a 'message' column.")
        else:
            statuses, matched_runbooks, confidences = [], [], []
            for _, row in scan_df.iterrows():
                log_text = f"[{row.get('level', 'INFO')}] {row.get('service', 'app')}: {row['message']}"
                top_hit = detector.retriever.retrieve(log_text, top_k=1)[0]
                is_normal = "Normal" in top_hit["title"]
                statuses.append("NORMAL" if is_normal else "ANOMALY")
                matched_runbooks.append(f"{top_hit['id']} — {top_hit['title']}")
                confidences.append(f"{top_hit['similarity_pct']}%")

            scan_df["Status"] = statuses
            scan_df["Matched Runbook"] = matched_runbooks
            scan_df["Match %"] = confidences

            total = len(scan_df)
            anomalies = (scan_df["Status"] == "ANOMALY").sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Logs Scanned", total)
            m2.metric("Normal Logs", total - anomalies)
            m3.metric("Anomalies Flagged", anomalies)

            st.bar_chart(scan_df["Status"].value_counts())
            st.dataframe(scan_df, use_container_width=True, hide_index=True)

            anomaly_list = [
                f"{r['service']} ({r['Matched Runbook']}): {r['message']}"
                for _, r in scan_df[scan_df["Status"] == "ANOMALY"].iterrows()
            ]
            if anomaly_list:
                st.subheader("⚡ Groq Executive Incident Summary")
                with st.spinner("Generating SRE summary with Groq..."):
                    st.markdown(detector.summarize_batch(anomaly_list, api_key=api_key or None))

# --- TAB 3: Add Custom Runbook ---
with tab3:
    st.subheader("Add a New Incident Runbook to ChromaDB")
    st.caption(f"Currently indexed runbooks: **{detector.retriever.collection.count()}**")

    with st.form("add_runbook_form", clear_on_submit=True):
        rb_title = st.text_input("Incident Title", placeholder="e.g., SSL Certificate Expiration")
        rb_category = st.selectbox(
            "Category",
            ["Database", "Infrastructure", "Security", "Networking", "Storage", "External API", "Baseline"],
        )
        rb_symptoms = st.text_area("Symptoms / Log Patterns", placeholder="e.g., TLS handshake error; x509: certificate has expired")
        rb_cause = st.text_area("Root Cause", placeholder="e.g., Automated certificate renewal failed.")
        rb_fix = st.text_area("Recommended Remediation", placeholder="e.g., Trigger cert-manager renewal and reload ingress.")

        if st.form_submit_button("Add to Knowledge Base", type="primary"):
            if rb_title.strip() and rb_symptoms.strip():
                new_id = detector.retriever.add_runbook(rb_title, rb_category, rb_symptoms, rb_cause, rb_fix)
                st.success(f"Added **{new_id} — {rb_title}** to ChromaDB! Test a matching log in Tab 1.")
            else:
                st.warning("Please fill in at least the Title and Symptoms.")