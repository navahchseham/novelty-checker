"""Provider-agnostic LLM access (see DESIGN.md §4).

Callers depend on the `LLM` protocol only. Add a new provider by writing a class with
`generate_json` and registering it in `get_llm`. Select it with the PROVIDER env var.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 6  # rounds through the model chain; backoff 5, 10, 20, 40, 80s between rounds
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


def rate_limit_wait(error: Exception) -> int:
    """Seconds to wait after a per-minute 429, taken from the API's retryDelay (default 60)."""
    m = re.search(r"retryDelay'?:\s*'(\d+)", str(error))
    return int(m.group(1)) + 1 if m else 60


class LLM(Protocol):
    last_model: str | None  # which model produced the most recent result

    def generate_json(self, prompt: str, schema: Any, temperature: float = 0.0) -> Any:
        """Return parsed JSON matching `schema` (a Pydantic model or list[Model])."""
        ...


class GeminiLLM:
    """Gemini adapter. Transient errors (429 / 5xx) move on to the next model in the chain;
    after the whole chain fails, it backs off exponentially and starts the chain again."""

    def __init__(self, model: str | None = None, fallbacks: str | None = None):
        from google import genai

        self._client = genai.Client()
        primary = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        if fallbacks is None:
            fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash")
        self.models = [primary] + [m.strip() for m in fallbacks.split(",") if m.strip() and m.strip() != primary]
        self.cache_id = f"gemini:{primary}"
        self.last_model: str | None = None

    def generate_json(self, prompt: str, schema: Any, temperature: float = 0.0) -> Any:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
        for attempt in range(MAX_RETRIES):
            for model in list(self.models):
                try:
                    resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
                    self.last_model = model
                    return json.loads(resp.text)
                except errors.APIError as e:
                    if e.code == 429 and "PerDay" in str(e):
                        # Daily quota exhausted: retrying only burns more quota, so drop the model for this run.
                        print(f"  {model}: 429 daily quota exhausted, removing from chain")
                        self.models.remove(model)
                    elif e.code == 429:
                        # Per-minute rate limit: wait as long as the API asks, then try again.
                        wait = rate_limit_wait(e)
                        print(f"  {model}: 429 rate limited, waiting {wait}s")
                        time.sleep(wait)
                    elif e.code >= 500:
                        print(f"  {model}: {e.code} {e.status}")
                    else:
                        raise
                except httpx.TransportError as e:
                    # Dropped connections and timeouts are transient: back off and retry.
                    print(f"  {model}: network error ({type(e).__name__})")
            if not self.models:
                raise RuntimeError("every model in the chain is out of quota")
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 5
                print(f"  all models busy, retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"all models unavailable after {MAX_RETRIES} rounds: {self.models}")


class CachedLLM:
    """Wraps any LLM with an on-disk cache keyed by (prompt, schema, temperature).

    Makes reruns free and repeatable, and lets an interrupted run resume.
    """

    def __init__(self, inner: LLM, cache_dir: str | Path = CACHE_DIR):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_model: str | None = None

    def _path(self, prompt: str, schema: Any, temperature: float) -> Path:
        model_id = getattr(self.inner, "cache_id", type(self.inner).__name__)
        key = hashlib.sha256(f"{model_id}|{schema!r}|{temperature}|{prompt}".encode()).hexdigest()
        return self.dir / f"{key}.json"

    def invalidate(self, prompt: str, schema: Any, temperature: float = 0.0) -> None:
        """Drop a cached result, e.g. one the caller rejected as invalid."""
        self._path(prompt, schema, temperature).unlink(missing_ok=True)

    def generate_json(self, prompt: str, schema: Any, temperature: float = 0.0) -> Any:
        path = self._path(prompt, schema, temperature)
        if path.exists():
            hit = json.loads(path.read_text())
            self.last_model = hit["model"]
            return hit["result"]
        result = self.inner.generate_json(prompt, schema, temperature)
        self.last_model = self.inner.last_model
        path.write_text(json.dumps({"model": self.last_model, "result": result}))
        return result


class GroqLLM:
    """Groq adapter (OpenAI-compatible chat API) with schema-constrained JSON output.

    429s wait for the server's retry-after; 5xx and network errors back off exponentially;
    a response that fails JSON validation is retried."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.cache_id = f"groq:{self.model}"
        self.last_model: str | None = None
        self._client = httpx.Client(timeout=180, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})

    def generate_json(self, prompt: str, schema: Any, temperature: float = 0.0) -> Any:
        from pydantic import TypeAdapter

        json_schema = TypeAdapter(schema).json_schema()
        wrapped = json_schema.get("type") == "array"  # the API wants an object at the top level
        if wrapped:
            json_schema = {"type": "object", "properties": {"items": json_schema}, "required": ["items"],
                           "$defs": json_schema.pop("$defs", {})}
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "response", "schema": json_schema, "strict": False}}}
        for attempt in range(MAX_RETRIES):
            try:
                r = self._client.post(self.URL, json=body)
            except httpx.TransportError as e:
                wait = 2 ** attempt * 5
                print(f"  {self.model}: network error ({type(e).__name__}), retrying in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                self.last_model = self.model
                out = json.loads(r.json()["choices"][0]["message"]["content"])
                return out["items"] if wrapped else out
            if r.status_code == 429 or r.status_code >= 500 or "json_validate_failed" in r.text:
                wait = float(r.headers.get("retry-after", 2 ** attempt * 5)) + 1
                print(f"  {self.model}: {r.status_code}, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
        raise RuntimeError(f"{self.model} unavailable after {MAX_RETRIES} attempts")


def get_llm(model: str | None = None, fallbacks: str | None = None, cache: bool = True,
            provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("PROVIDER", "gemini")).lower()
    if provider == "gemini":
        llm: LLM = GeminiLLM(model, fallbacks)
    elif provider == "groq":
        llm = GroqLLM(model)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return CachedLLM(llm) if cache else llm
