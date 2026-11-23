"""
Evaluation Script for RAG Retrieval Accuracy
Run with: python eval.py
"""

from src.retriever import RunbookRetriever

# Labeled test cases using paraphrased/unseen logs (not copied from runbooks.json)
TEST_CASES = [
    {
        "log": "[ERROR] api-server: TimeoutError acquiring DB connection from pool",
        "expected_id": "RB-001",
    },
    {
        "log": "[CRITICAL] worker-pod: Process killed due to exceeding 4GiB RAM limit",
        "expected_id": "RB-002",
    },
    {
        "log": "[WARN] web-auth: User locked out after 65 failed password attempts from 45.33.12.1",
        "expected_id": "RB-003",
    },
    {
        "log": "[INFO] gateway: GET /health 200 OK in 12ms",
        "expected_id": "RB-004",
    },
    {
        "log": "[ERROR] postgres-proxy: no remaining connection slots available for clients",
        "expected_id": "RB-001",
    },
    {
        "log": "[INFO] cache: Redis hit rate steady at 96% after scheduled job finished",
        "expected_id": "RB-004",
    },
]


def run_eval():
    retriever = RunbookRetriever()
    top1_hits = 0
    top2_hits = 0

    print("\n--- RAG Retrieval Evaluation ---\n")
    for case in TEST_CASES:
        results = retriever.retrieve(case["log"], top_k=2)
        retrieved_ids = [r["id"] for r in results]

        is_top1 = retrieved_ids[0] == case["expected_id"]
        is_top2 = case["expected_id"] in retrieved_ids

        top1_hits += int(is_top1)
        top2_hits += int(is_top2)

        status = "PASS" if is_top1 else ("TOP-2" if is_top2 else "MISS")
        print(f"[{status}] Expected: {case['expected_id']} | Got: {retrieved_ids} | Log: {case['log'][:55]}...")

    total = len(TEST_CASES)
    print("\n--- Summary ---")
    print(f"Top-1 Accuracy (Hit@1): {top1_hits}/{total} ({top1_hits / total * 100:.1f}%)")
    print(f"Top-2 Accuracy (Hit@2): {top2_hits}/{total} ({top2_hits / total * 100:.1f}%)\n")


if __name__ == "__main__":
    run_eval()