"""Benchmark retrieval methods on dev and test splits."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from starter.baseline import pack_context, hit, TfidfIndex


def run_benchmark(chunks_path="data/chunks.json", questions_path="data/questions.json"):
    chunks = json.loads(Path(chunks_path).read_text(encoding='utf-8'))
    questions = json.loads(Path(questions_path).read_text(encoding='utf-8'))
    
    # Initialize retrievers
    print("Initializing retrievers...")
    tfidf = TfidfIndex(chunks)
    bm25 = LexicalRetriever(chunks)
    dense = DenseRetriever(qdrant_path="data/qdrant", collection_name="python_tutorial_C")
    hybrid = HybridRetriever(dense, bm25)
    
    configs = [
        {
            'name': 'TF-IDF Baseline',
            'retriever': 'tfidf',
            'top_k': 3,
            'budget': 800,
            'use_filter': False,
        },
        {
            'name': 'Method 1: Dense (Qdrant)',
            'retriever': 'dense',
            'top_k': 3,
            'budget': 800,
            'use_filter': False,
        },
        {
            'name': 'Method 2: BM25 Lexical',
            'retriever': 'bm25',
            'top_k': 3,
            'budget': 800,
            'use_filter': False,
        },
        {
            'name': 'Method 3: Hybrid (RRF)',
            'retriever': 'hybrid',
            'top_k': 3,
            'budget': 800,
            'use_filter': False,
        },
        {
            'name': 'Cải thiện 1: Hybrid + Section Filter',
            'retriever': 'hybrid',
            'top_k': 3,
            'budget': 800,
            'use_filter': True,
        },
        {
            'name': 'Cải thiện 2 (Chốt): Hybrid + Filter + Budget 1600',
            'retriever': 'hybrid',
            'top_k': 3,
            'budget': 1600,
            'use_filter': True,
        },
    ]

    all_results = []

    for split in ['dev', 'test']:
        split_questions = [q for q in questions if q['split'] == split]
        print(f"\n=================== Split: {split.upper()} (N={len(split_questions)}) ===================")
        
        for cfg in configs:
            name = cfg['name']
            retriever_type = cfg['retriever']
            top_k = cfg['top_k']
            budget = cfg['budget']
            use_filter = cfg['use_filter']
            
            latencies = []
            hit1_count = 0
            hit3_count = 0
            context_hit_count = 0
            traces = []
            
            for q in split_questions:
                q_text = q['question']
                gold = q['gold']
                filters = q.get('filter') if use_filter else None
                
                t0 = time.perf_counter()
                if retriever_type == 'tfidf':
                    # TF-IDF from starter doesn't take filters
                    results = tfidf.search(q_text, top_k=top_k)
                elif retriever_type == 'bm25':
                    results = bm25.search(q_text, filters=filters, top_k=top_k)
                elif retriever_type == 'dense':
                    results = dense.search(q_text, filters=filters, top_k=top_k)
                elif retriever_type == 'hybrid':
                    results = hybrid.search(q_text, filters=filters, top_k=top_k)
                else:
                    raise ValueError(f"Unknown retriever: {retriever_type}")
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)
                
                # Check hits
                top1_hit = hit(results[0], gold) if results else False
                top3_hit = any(hit(c, gold) for c in results[:3])
                
                context, ids = pack_context(results, budget=budget)
                context_chunks = [c for c in results if c['id'] in ids]
                ctx_hit = any(hit(c, gold) for c in context_chunks)
                
                if top1_hit:
                    hit1_count += 1
                if top3_hit:
                    hit3_count += 1
                if ctx_hit:
                    context_hit_count += 1
                    
                traces.append({
                    'query_id': q['id'],
                    'question': q_text,
                    'hit1': top1_hit,
                    'hit3': top3_hit,
                    'context_hit': ctx_hit,
                    'results': [{'id': c['id'], 'source': c['source_id'], 'page': c['page'], 'score': c.get('score')} for c in results],
                    'context_ids': ids,
                    'latency_ms': elapsed_ms
                })
                
            n = len(split_questions)
            med_ms = statistics.median(latencies)
            res_summary = {
                'config': name,
                'split': split,
                'N': n,
                'hit1': hit1_count,
                'hit1_rate': f"{hit1_count}/{n} ({hit1_count/n*100:.1f}%)",
                'hit3': hit3_count,
                'hit3_rate': f"{hit3_count}/{n} ({hit3_count/n*100:.1f}%)",
                'context_hit': context_hit_count,
                'context_rate': f"{context_hit_count}/{n} ({context_hit_count/n*100:.1f}%)",
                'median_ms': round(med_ms, 2),
                'cost_usd': 0.0
            }
            all_results.append((res_summary, traces))
            print(f"[{split.upper()}] {name:50s} | Hit@1: {res_summary['hit1_rate']:14s} | Hit@3: {res_summary['hit3_rate']:14s} | CtxHit: {res_summary['context_rate']:14s} | {med_ms:.2f}ms")

    # Save summary and traces
    out_dir = Path("results/retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_list = [r[0] for r in all_results]
    (out_dir / "summary.json").write_text(json.dumps(summary_list, ensure_ascii=False, indent=2), encoding='utf-8')
    for summary, traces in all_results:
        slug = f"{summary['split']}_{summary['config'].replace(':', '').replace('/', '_').replace(' ', '_')}"
        (out_dir / f"traces_{slug}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
        
    print(f"\nAll retrieval benchmarks completed and saved to {out_dir}/")


if __name__ == '__main__':
    run_benchmark()
