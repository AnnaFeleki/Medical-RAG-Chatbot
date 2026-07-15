"""Tiny retrieval evaluation harness.

Measures whether the correct source document is retrieved for a set of
questions. Run AFTER indexing the documents referenced in your questions file.

    python eval.py --questions eval/questions.example.json --k 4

Each question item is: {"question": "...", "expected_source": "file.pdf"}
Reports Hit-rate@k (was the right file in the top-k?) and MRR@k.
"""

import argparse
import json
import os

from rag_chain import get_vectorstore


def evaluate(questions_path: str, persist_dir: str = "chroma_db", k: int = 4) -> dict:
    if not os.path.isdir(persist_dir):
        raise SystemExit(f"No index at {persist_dir}. Index your PDFs first.")
    with open(questions_path) as f:
        items = json.load(f)

    vs = get_vectorstore(persist_dir)
    hits, reciprocal = 0, 0.0

    for item in items:
        docs = vs.similarity_search(item["question"], k=k)
        retrieved = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        expected = item["expected_source"]
        if expected in retrieved:
            hits += 1
            reciprocal += 1.0 / (retrieved.index(expected) + 1)
        status = "✓" if expected in retrieved else "✗"
        print(f"  {status} {item['question'][:60]:60s} -> {retrieved}")

    n = len(items)
    result = {"n": n, "hit_rate@k": hits / n, "mrr@k": reciprocal / n, "k": k}
    print(f"\nHit-rate@{k}: {result['hit_rate@k']:.2f}   MRR@{k}: {result['mrr@k']:.2f}   (n={n})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--questions", default="eval/questions.example.json")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.questions, args.persist_dir, args.k)
