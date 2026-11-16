import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrievers import LexicalRetriever, HybridRetriever


class MockDense:
    def search(self, query, filters=None, top_k=3):
        return [
            {'id': 'c1', 'source_id': 's1', 'page': 1, 'text': 'python range stop value', 'metadata': {'section': 'controlflow'}, 'score': 0.9, 'strategy': 'dense'},
            {'id': 'c2', 'source_id': 's1', 'page': 2, 'text': 'break statement in loops', 'metadata': {'section': 'controlflow'}, 'score': 0.8, 'strategy': 'dense'},
        ][:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {'id': 'c1', 'source_id': 's1', 'source': 'http://s1', 'page': 1, 'text': 'The range function stop value is never part of sequence.', 'metadata': {'section': 'controlflow'}},
            {'id': 'c2', 'source_id': 's1', 'source': 'http://s1', 'page': 2, 'text': 'The break statement breaks out of innermost loop.', 'metadata': {'section': 'controlflow'}},
            {'id': 'c3', 'source_id': 's2', 'source': 'http://s2', 'page': 1, 'text': 'Tuples are immutable sequences in Python.', 'metadata': {'section': 'datastructures'}},
        ]
        self.lexical = LexicalRetriever(self.chunks)

    def test_bm25_retrieval(self):
        hits = self.lexical.search("range stop value", top_k=2)
        self.assertTrue(len(hits) > 0)
        self.assertEqual(hits[0]['id'], 'c1')
        self.assertEqual(hits[0]['strategy'], 'lexical_bm25')

    def test_bm25_with_filter(self):
        hits = self.lexical.search("statement", filters={'section': 'datastructures'}, top_k=2)
        self.assertEqual(len(hits), 0)
        hits2 = self.lexical.search("immutable", filters={'section': 'datastructures'}, top_k=2)
        self.assertEqual(len(hits2), 1)
        self.assertEqual(hits2[0]['id'], 'c3')

    def test_hybrid_rrf(self):
        hybrid = HybridRetriever(MockDense(), self.lexical)
        hits = hybrid.search("range stop value", top_k=2)
        self.assertTrue(len(hits) > 0)
        self.assertEqual(hits[0]['id'], 'c1')
        self.assertEqual(hits[0]['strategy'], 'hybrid_rrf')


if __name__ == '__main__':
    unittest.main()
