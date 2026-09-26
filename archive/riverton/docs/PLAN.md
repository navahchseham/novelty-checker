# Build Plan: Rewarding Novelty in Submissions (Problem Statement 3)

This file lists the build steps and tracks progress. Each step is built, tested and
documented before the next one starts.

## Requirements (from the problem statement)

| # | Requirement | Covered by step |
|---|-------------|-----------------|
| T1 | Define the "shape" of the user content: **3 discrete user-provided properties**, submitted against fixed content of **≤ 100 words** | 1 |
| T2 | Describe how novelty is rewarded relative to other submissions while keeping **a minimum level of relevance** | 2 |
| T3 | Implement novelty evaluation of a new submission against **≈ 50 other submissions** (LLM-generated synthetic data recommended) | 3–7 |
| T4 | Automated tests showing: novel content is rewarded, non-novel content is not, and **high-novelty but low-relevance content is not rewarded** | 8 |
| — | Output is a score normalized to **[0.0, 1.0]** | 6–7 |
| S1 | Write-up: design, rationale, success criteria, level of achievement (Markdown) | 9–10 |
| S2 | Disclosure of coding-agent use and how it was directed | ongoing, `docs/AGENT_COLLABORATION.md` |
| S3 | GitHub repo containing code, golden dataset and tests | 10 |

## Steps

| Step | Name | Output | Status |
|------|------|--------|--------|
| 0 | Project setup and documentation structure | folder layout, `requirements.txt`, `.gitignore`, docs skeleton | ✅ Done |
| 1 | Content shape: fixed content and submission schema | `novelty/schema.py`, `data/fixed_content.json`, `tests/test_schema.py`, DESIGN §1 | ✅ Done |
| 2 | Scoring design (relevance and quality gates, embedding and LLM-rubric novelty, normalization) and success criteria | DESIGN §2 & §5, `novelty/scoring.py`, `tests/test_scoring.py` | ✅ Done |
| 3 | Synthetic dataset: angle plan, ~50 corpus submissions, labeled golden test set | `data/angles.json`, `scripts/generate_data.py`, `data/corpus.json`, `data/golden.json`, `tests/test_data.py` | ✅ Done |
| 4 | Embedding layer (Gemini embeddings with an on-disk cache) and diagnostics | `novelty/embeddings.py`, `tests/test_embeddings.py`, `scripts/inspect_embeddings.py`, DESIGN §4.1 | ✅ Done |
| 5 | LLM rubric judge (relevance, quality and novelty levels, JSON output, cached) | `novelty/judge.py` | ⏳ Next |
| 6 | Embedding novelty (top-k retrieval, leave-one-out ECDF normalization) | `novelty/novelty.py` | ☐ |
| 7 | End-to-end pipeline and CLI (`score a new submission`) | `novelty/pipeline.py`, `scripts/score.py` | ☐ |
| 8 | Automated tests: unit tests (offline fake embedder) and golden-set tests | `tests/` | ☐ |
| 9 | Evaluation run: calibration, metrics against success criteria | `docs/EVALUATION.md` | ☐ |
| 10 | Final write-up, agent disclosure, push to GitHub | `README.md`, `docs/*` | ☐ |

Optional extension if time allows: an LLM "idea-level" novelty judge (Gemini) that
pulls out the key claims of a submission and checks them against the claims in the corpus.
Its score would be blended with the embedding score.

## Progress log

- **2026-09-26: Step 0.** Read the problem statements and chose PS3. Created the
  folder layout, `.gitignore` (keeps `.env` and `myvenv/` out of git), `requirements.txt`
  and the documentation skeleton. Added `numpy` and `pytest` to the venv. The Gemini
  API key was already confirmed working via `test.py`.
- **2026-09-26: Repo move.** The project moved into `hack/`, a clone of
  `github.com/navahchseham/hack`. It is now the repo root. The venv stays one level up
  (`../myvenv`).
- **2026-09-26: Step 1.** Design discussion: we agreed novelty is measured against peer
  submissions, not the web. It is idea-level, not plagiarism detection. The pipeline
  combines semantic search with an LLM rubric judge. Built the content shape: a fictional
  78-word news item (Riverton four-day week pilot) and a `Submission` model with
  `headline` / `body` / `stance`. 9 schema tests pass
  (`../myvenv/bin/python -m pytest`).
- **2026-09-26: Step 2.** Wrote the detailed scoring design (DESIGN §2):
  `score = (0.4·N_emb + 0.6·N_llm) × gate(R) × gate(Q)`. N_emb is the percentile of the
  distance to the closest neighbour and the top-5 neighbours. One LLM judge call returns
  anchored 0–4 levels for relevance, quality and novelty. The gates are multiplicative,
  so off-topic content and junk get exactly 0. Set success criteria SC1–SC8 before
  building (DESIGN §5). Implemented the math as pure functions in `novelty/scoring.py`
  with 9 unit tests (18 tests in total pass). Revised steps 5 and 6 to match the design.
- **2026-09-26: Step 3 (part A).** Wrote the angle plan `data/angles.json`: 13 corpus
  angles (27 popular / 14 medium / 5 rare / 4 generic = 50 items), 6 held-back angles, a
  golden plan of 26 items across 7 categories, and 5 hand-written junk items. Wrote
  `novelty/llm.py` (the `LLM` protocol and a Gemini adapter, selected with `PROVIDER`)
  and `scripts/generate_data.py` (seeded persona/style/length slots, schema validation,
  retries, `--dry-run`). The dry run looks correct. No API calls yet; waiting for the
  developer to review the angle plan.
- **2026-09-26: Step 3 (part B), generation run.** The first run failed:
  `gemini-2.5-flash` returned 404 ("no longer available to new users"). Listed the
  available models and switched the default to `gemini-3.8-flash`. The next runs got
  persistent 503 "high demand" errors. Fixes in `novelty/llm.py`: a fallback model chain
  with exponential backoff, and the `CachedLLM` disk cache so interrupted runs resume. A
  cached result that fails schema validation is evicted before the retry. Each item now
  records the `model` that wrote it.
- **2026-09-26: Step 3 (part C), dataset done.** `gemini-3.8-flash` hit its free-tier
  *daily* quota, and retrying 429s had used up more of it. Regenerated **everything** with
  one model (`gemini-3.5-flash-lite`) so writing style can't mix with ideas. That run hit
  the 15 requests/minute limit, so the adapter now tells per-minute limits (wait for
  `retryDelay`) apart from per-day limits (remove the model). The run resumed from cache.
  Result: 50 corpus + 26 golden items. Read all golden items by hand: every one matches
  its angle. A leakage scan found no held-back ideas in the corpus. Added
  `tests/test_data.py`. **100 tests pass.**
- **2026-09-26: Step 4.** Built `novelty/embeddings.py`: an `Embedder` protocol,
  `GeminiEmbedder` (`gemini-embedding-2`, chosen over `001` for a 0.29 vs 0.17
  paraphrase/off-topic gap), a per-text disk cache, and an offline `FakeEmbedder`. Found
  and fixed a silent bug: `embedding-2` merges a list of strings into one vector unless
  each text is wrapped in its own `Content`. Diagnostics on the real data: 87%
  nearest-neighbour angle accuracy; novel items N_emb ≥ 0.78; but one common paraphrase
  reaches 0.76; junk gets 1.00 (the gates are needed); and embedding relevance cannot
  separate the bike-lane hard negative (0.681 vs 0.683). This supports using the LLM
  judge. 105 tests pass.
