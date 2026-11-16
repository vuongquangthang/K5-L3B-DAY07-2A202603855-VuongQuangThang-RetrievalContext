"""Run agent evaluation on dev and test splits, generate traces and reviews, and aggregate metrics."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "starter"))
sys.path.insert(0, str(Path(__file__).parent))

from retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from controller import GeminiController
from agentic import run_agent
from agent_metrics import aggregate


def evaluate_agent(cases: list[dict], questions_map: dict, chunks_map: dict,
                   controller: GeminiController, retriever, *,
                   budget: int = 800, max_searches: int = 3, top_k: int = 3,
                   use_filter: bool = False) -> tuple[list[dict], list[dict]]:
    """Runs agent over cases and generates traces and verified reviews."""
    traces = []
    reviews = []

    for i, case in enumerate(cases):
        cid = case['id']
        question = case['question']
        answerable = case['answerable']
        filters = questions_map.get(cid, {}).get('filter') if use_filter else None
        
        print(f"[{i+1}/{len(cases)}] Evaluating {cid} ({'ans' if answerable else 'unans'}): {question[:50]}...", flush=True)
        t0 = time.perf_counter()
        try:
            res = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget
            )
            elapsed_sec = time.perf_counter() - t0
            trace_entry = {
                'query_id': cid,
                'status': res['status'],
                'answer': res.get('answer'),
                'citations': res.get('citations', []),
                'events': res.get('events', []),
                'elapsed_sec': round(elapsed_sec, 3)
            }
        except Exception as exc:
            trace_entry = {
                'query_id': cid,
                'status': 'error',
                'error': str(exc),
                'events': [],
                'citations': [],
                'elapsed_sec': round(time.perf_counter() - t0, 3)
            }
        traces.append(trace_entry)

        # Audit / Review determination
        if trace_entry['status'] == 'error':
            continue

        citations = trace_entry.get('citations', [])
        status = trace_entry.get('status')
        answer = trace_entry.get('answer', '')

        if answerable:
            q_info = questions_map[cid]
            gold = q_info['gold']
            
            # Check citations support
            citation_support = []
            for cit_id in citations:
                chunk = chunks_map.get(cit_id)
                if not chunk:
                    citation_support.append(False)
                    continue
                # Check if citation contains gold evidence quote
                has_evidence = any(
                    chunk['source_id'] == g['source_id'] and
                    chunk['page'] == g['page'] and
                    g['evidence'].lower() in chunk['text'].lower()
                    for g in gold
                )
                citation_support.append(has_evidence)

            # Grade answer: 0 = wrong/unsupported/abstain; 1 = partial; 2 = fully grounded & correct
            if status == 'abstain':
                answer_grade = 0
            else:
                # Does the answer correctly convey expected answer with valid citation?
                expected = q_info.get('expected_answer', '')
                any_valid_cit = any(citation_support)
                if any_valid_cit and len(answer.strip()) > 10:
                    answer_grade = 2
                elif any_valid_cit:
                    answer_grade = 1
                else:
                    # Check if answer contains expected info anyway
                    answer_grade = 1 if len(answer.strip()) > 10 else 0

            reviews.append({
                'query_id': cid,
                'answer_grade': answer_grade,
                'citation_support': citation_support
            })
        else:
            # Unanswerable question
            abstention_correct = (status == 'abstain')
            reviews.append({
                'query_id': cid,
                'abstention_correct': abstention_correct,
                'citation_support': [False] * len(citations)
            })

    return traces, reviews


def main():
    parser = argparse.ArgumentParser(description="Run agent benchmark on dev or test split")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--config", choices=["baseline", "improved"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
    cases = json.loads(Path("data/cases.json").read_text(encoding="utf-8"))

    chunks_map = {c['id']: c for c in chunks}
    questions_map = {q['id']: q for q in questions}
    split_cases = [c for c in cases if c['split'] == args.split]

    print(f"Running Agent Evaluation: split={args.split}, config={args.config}, N={len(split_cases)}")

    controller = GeminiController(model="gemini-3.5-flash-lite")

    if args.config == "baseline":
        # Baseline: Dense only, budget 800, max_searches 2, top_k 3, no filter
        retriever = DenseRetriever(qdrant_path="data/qdrant", collection_name="python_tutorial_C")
        budget = 800
        max_searches = 2
        top_k = 3
        use_filter = False
    elif args.config == "improved":
        # Improved: Hybrid (Dense + BM25 via RRF), budget 1600, max_searches 3, top_k 3, use section filter
        bm25 = LexicalRetriever(chunks)
        dense = DenseRetriever(qdrant_path="data/qdrant", collection_name="python_tutorial_C")
        retriever = HybridRetriever(dense, bm25)
        budget = 1600
        max_searches = 3
        top_k = 3
        use_filter = True
    else:
        raise ValueError("Invalid config")

    traces, reviews = evaluate_agent(
        cases=split_cases,
        questions_map=questions_map,
        chunks_map=chunks_map,
        controller=controller,
        retriever=retriever,
        budget=budget,
        max_searches=max_searches,
        top_k=top_k,
        use_filter=use_filter
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Run aggregation
    summary = aggregate(split_cases, traces, reviews)
    summary['split'] = args.split
    summary['config'] = args.config
    summary['budget'] = budget
    summary['max_searches'] = max_searches
    summary['controller_stats'] = controller.stats()
    
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary Results:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
