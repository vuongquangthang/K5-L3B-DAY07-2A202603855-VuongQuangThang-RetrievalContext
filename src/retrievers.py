"""Retrieval methods for Day 7: Dense (Qdrant + FastEmbed), Lexical (BM25/TF-IDF), and Hybrid (RRF)."""
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class LexicalRetriever:
    """BM25 Lexical Retriever."""
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        
        # Inverted index & DF
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1
                
        # IDF
        n_docs = len(chunks)
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
            
        scores = []
        for idx, chunk in enumerate(self.chunks):
            # Check filters
            if filters:
                match = True
                for fk, fv in filters.items():
                    if chunk.get('metadata', {}).get(fk) != fv and chunk.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue
                    
            tf = self.term_freqs[idx]
            dl = self.doc_lens[idx]
            score = 0.0
            for t in q_tokens:
                if t in tf:
                    t_freq = tf[t]
                    idf = self.idf.get(t, 0.0)
                    num = t_freq * (self.k1 + 1)
                    denom = t_freq + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                    score += idf * (num / denom)
            if score > 0:
                scores.append((score, chunk))
                
        scores.sort(key=lambda x: (-x[0], x[1]['id']))
        results = []
        for score, chunk in scores[:top_k]:
            results.append({
                **chunk,
                'score': float(score),
                'strategy': 'lexical_bm25'
            })
        return results


class DenseRetriever:
    """Dense Retriever using Day 6 Qdrant persistent vector store & FastEmbed all-MiniLM-L6-v2."""
    def __init__(self, qdrant_path: str = "data/qdrant", collection_name: str = "python_tutorial_C",
                 cache_dir: str | Path | None = None):
        os.environ['HF_HUB_OFFLINE'] = '1'
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # Load FastEmbed model offline from local cache
        model_path = Path('.cache/models/models--qdrant--all-MiniLM-L6-v2-onnx/snapshots/d13954661f83248295ba75c1ed411eef3b7b936e').resolve()
        if not model_path.exists():
            # Check alternative cache path
            alt = Path('D:/vuongquangthang/K5-L3B-DAY06-VuongQuangThang-2A202603855-DataIngestion/.cache/models/models--qdrant--all-MiniLM-L6-v2-onnx/snapshots/d13954661f83248295ba75c1ed411eef3b7b936e').resolve()
            if alt.exists():
                model_path = alt
                
        self.embedder = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            specific_model_path=str(model_path) if model_path.exists() else None,
            threads=4,
            providers=["CPUExecutionProvider"]
        )
        self.client = QdrantClient(path=self.qdrant_path)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        
        vector = list(self.embedder.embed([query]))[0].tolist()
        
        # Build Qdrant filter
        qfilter = None
        if filters:
            must_clauses = []
            for fk, fv in filters.items():
                must_clauses.append(
                    qmodels.FieldCondition(
                        key=fk,
                        match=qmodels.MatchValue(value=fv)
                    )
                )
            if must_clauses:
                qfilter = qmodels.Filter(must=must_clauses)
                
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=qfilter,
            limit=top_k
        )
            
        results = []
        for h in hits:
            pl = h.payload
            results.append({
                'id': str(h.id),
                'source_id': pl.get('document_id', pl.get('source_id')),
                'source': pl.get('source'),
                'page': pl.get('page'),
                'text': pl.get('text'),
                'metadata': {
                    'section': pl.get('section'),
                    'version': pl.get('version'),
                    'language': pl.get('language'),
                    'chunking': pl.get('chunking'),
                    'start': pl.get('start'),
                    'end': pl.get('end'),
                },
                'score': float(h.score),
                'strategy': 'dense'
            })
        return results


class HybridRetriever:
    """Hybrid Retriever combining Dense (Qdrant) and Lexical (BM25) via Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, lexical_retriever: LexicalRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3,
               candidate_pool: int = 15) -> list[dict[str, Any]]:
        pool_k = max(top_k * 3, candidate_pool)
        dense_hits = self.dense.search(query, filters=filters, top_k=pool_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=pool_k)
        
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        
        # RRF for dense
        for rank, hit in enumerate(dense_hits):
            cid = hit['id']
            chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        # RRF for lexical
        for rank, hit in enumerate(lexical_hits):
            cid = hit['id']
            if cid not in chunk_map:
                chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        sorted_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            c = dict(chunk_map[cid])
            c['score'] = float(scores[cid])
            c['strategy'] = 'hybrid_rrf'
            results.append(c)
        return results
