"""Gemini-powered controller for Agentic RAG."""
import json
import os
import re
import time
from typing import Any
import httpx


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # Try reading from known env files
    for candidate in [
        r'D:\vuongquangthang\VinUni_Codelab_Day01_Template\.env',
        r'D:\vuongquangthang\K5-L3B-DAY04-VuongQuangThang-2A202603855-ReactLoopAgent\.env',
        '.env'
    ]:
        if os.path.exists(candidate):
            with open(candidate, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GEMINI_API_KEY='):
                        return line.split('=', 1)[1].strip().strip('"\'')
    raise ValueError("GEMINI_API_KEY not found in environment or .env files.")


SYSTEM_PROMPT = """You are an accurate, honest AI retrieval assistant answering questions based STRICTLY on retrieved documentation chunks.

You must output a single JSON object with your next action:

Option 1: Search the documentation. Use this when no context exists yet, or when the current context is insufficient to answer the question:
{"action": "search", "query": "<focused search keywords>"}

Option 2: Answer the question when the context contains explicit, verifiable evidence:
{"action": "answer", "answer": "<concise answer directly supported by context [chunk-id]>", "citations": ["<chunk-id>"]}

Option 3: Abstain when the retrieved context does not contain enough evidence and the topic is absent:
{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

STRICT CITATION RULES:
1. Every ID in "citations" must be a real chunk ID copied exactly from the context header (e.g. [0296acff-... | ...]).
2. Every cited ID must appear in square brackets inside the answer text, e.g. "The range end point is excluded [cf7be23a-7374-59e7-959c-f540f33b4073]."
3. Never invent facts or citations. If evidence is missing, abstain.
4. Output ONLY valid JSON, with no surrounding prose or markdown ticks.
"""


class GeminiController:
    """Controller implementation satisfying the agentic.Controller Protocol."""
    def __init__(self, model: str = "gemini-3.5-flash-lite", timeout: float = 30.0):
        self.api_key = get_api_key()
        self.model = model
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_sec = 0.0
        self.call_count = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide next action (search or answer) based on question and history."""
        self.call_count += 1
        
        user_msg = f"Question: {question}\n\n"
        if history:
            last_entry = history[-1]
            queries = last_entry.get('queries', [])
            context = last_entry.get('context', '')
            user_msg += f"Past search queries: {queries}\n\n"
            user_msg += f"Retrieved Context:\n{context}\n\n"
            user_msg += "Does the retrieved context contain explicit evidence to answer the question? If yes, answer with citations. If no and more searches may help, search. If the topic is completely absent from the documentation, abstain."
        else:
            user_msg += "No context retrieved yet. Please generate a search query."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            'contents': [
                {'role': 'user', 'parts': [{'text': SYSTEM_PROMPT + '\n\n' + user_msg}]}
            ],
            'generationConfig': {
                'responseMimeType': 'application/json',
                'temperature': 0.0
            }
        }

        t0 = time.perf_counter()
        # Retry up to 3 times with exponential backoff & fallback model on 503
        last_exc = None
        models_to_try = [self.model, "gemini-3.1-flash-lite", self.model]
        for attempt, current_model in enumerate(models_to_try):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self.api_key}"
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 429:
                        time.sleep(3.0 * (attempt + 1))
                        raise RuntimeError(f"Rate limited (429): {resp.text[:150]}")
                    if resp.status_code == 503:
                        time.sleep(2.0 * (attempt + 1))
                        raise RuntimeError(f"Model unavailable (503): {resp.text[:150]}")
                    if resp.status_code != 200:
                        raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:200]}")
                    data = resp.json()
                    
                    # Track tokens
                    usage = data.get('usageMetadata', {})
                    self.total_prompt_tokens += usage.get('promptTokenCount', 0)
                    self.total_completion_tokens += usage.get('candidatesTokenCount', 0)
                    
                    text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                    break
            except Exception as e:
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Gemini API failed after attempts: {last_exc}")

        self.total_latency_sec += (time.perf_counter() - t0)

        # Parse JSON
        try:
            # Strip markdown formatting if any
            clean_text = text
            if clean_text.startswith('```'):
                clean_text = re.sub(r'^```(?:json)?\s*', '', clean_text)
                clean_text = re.sub(r'\s*```$', '', clean_text)
            parsed = json.loads(clean_text)
            if not isinstance(parsed, dict) or 'action' not in parsed:
                raise ValueError("Response missing 'action'")
            return parsed
        except Exception:
            # Fallback heuristic
            if not history:
                return {'action': 'search', 'query': question}
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}

    def stats(self) -> dict[str, Any]:
        # Pricing for Gemini 3.5 Flash Lite: ~$0.075 / 1M input, $0.30 / 1M output
        cost = (self.total_prompt_tokens / 1_000_000 * 0.075) + (self.total_completion_tokens / 1_000_000 * 0.30)
        return {
            'model': self.model,
            'calls': self.call_count,
            'total_latency_sec': round(self.total_latency_sec, 3),
            'avg_latency_sec': round(self.total_latency_sec / max(self.call_count, 1), 3),
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'estimated_cost_usd': round(cost, 6)
        }
