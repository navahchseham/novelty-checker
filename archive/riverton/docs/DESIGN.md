# Design: Novelty Scoring Pipeline

> This is a working document. Each section is filled in when its step in [PLAN.md](PLAN.md) is built.

## 1. Content shape

**Scenario:** readers comment on a short local-news item.

### Fixed content ([data/fixed_content.json](../data/fixed_content.json), 78 words)
> **Riverton approves four-day work week pilot for city staff.** The Riverton City Council
> voted 7-2 on Tuesday to launch a one-year pilot of a four-day work week for its 1,200
> municipal employees. […] The city will publish productivity and resident-satisfaction
> data every quarter.

The story is fictional, so nothing is factually wrong and the model has no prior
knowledge of it. We picked this topic because it invites many different angles:
economics, childcare, public services, climate, labour, and ways to measure the pilot.
That makes some takes common and others rare, which is what novelty scoring needs.

### Submission: 3 user-provided properties ([novelty/schema.py](../novelty/schema.py))

| Property | Type | Constraint | Purpose |
|----------|------|-----------|---------|
| `headline` | free text | 1–20 words | Short summary of the user's point |
| `body` | free text | 10–250 words | The argument itself: the main place new ideas appear |
| `stance` | multi-choice | `support` \| `oppose` \| `mixed` \| `question` | Structured signal of the user's position |

`id` is optional and is set by the system, not by the user.

### Design choices
- **Validation at the boundary.** Pydantic rejects malformed input before scoring, and whitespace is trimmed.
  The **10-word minimum body** rejects empty input such as "lol" at the schema level. This
  does not decide low-effort or off-topic content; that is the job of the scoring gates (§2).
- **Stance is not embedded.** `Submission.to_text()` returns only `headline + body`. If
  "Stance: support" were part of every embedded text, all submissions with the same stance
  would look more alike than they are, and the novelty signal would be skewed. Stance is
  still available as separate data and is passed to the LLM judge as context.

## 2. Scoring approach

### 2.1 Overview
```
new submission x
  │
  ├─ embed(x) ──► cosine sims to corpus ──► top-k neighbours ──► N_emb  (embedding novelty)
  │                                               │
  └─ LLM judge (one call, JSON output) ◄──────────┘  sees: fixed content, x (with stance), top-k texts
        ├─ relevance level  R ∈ {0..4}
        ├─ quality level    Q ∈ {0..4}
        └─ novelty level    L ∈ {0..4} ──► N_llm = L / 4

  novelty = 0.4·N_emb + 0.6·N_llm
  score   = novelty × gate(R) × gate(Q)        ∈ [0, 1]
```
The math is implemented in [novelty/scoring.py](../novelty/scoring.py) (pure functions, fully
unit-tested). All weights are in `ScoringConfig`. The values below are **starting values**
that Step 9 will check against the golden set.

### 2.2 Embedding novelty `N_emb`
For a submission `x` with cosine similarities `s_1..s_n` to the corpus items:

1. **Raw distance:** `d(x) = 1 − (0.5·max(s) + 0.5·mean(top-5 s))`
   - The **max** term catches a near-duplicate of any single item.
   - The **top-k mean** term catches a *crowded* point. If five people already said it,
     the idea is well covered even if nobody said it word for word.
2. **Normalization:** `N_emb = ECDF(d(x))` against the reference distribution
   `{d(c) : c ∈ corpus}`. Each corpus item is scored leave-one-out against the rest.

**Why normalize by percentile instead of using raw cosine?** Embedding models squeeze
similarities into a narrow band (e.g. 0.55–0.90), and that band differs between models.
The percentile answers "how unusual is this compared with a typical submission here?" and
stays valid if the embedding provider changes. It also fits the corpus we have: in a
corpus full of near-copies, a moderately different submission counts as novel.

### 2.3 LLM rubric judge
There is **one call per submission**. The judge sees the fixed content, the submission
(headline, body, stance) and the **top-5 retrieved neighbours**. It returns three integer
levels as JSON:

| Level | Relevance `R` | Quality `Q` | Novelty `L` (vs. shown neighbours) |
|---|---|---|---|
| 0 | Unrelated to the story | Gibberish / spam | Restates an existing point |
| 1 | Mentions the topic only in passing, or is generic | Very low effort, no reasoning | Existing point with trivial rewording or detail |
| 2 | About four-day weeks in general, not this story | Clear claim, thin reasoning | Existing point plus a meaningful new detail or example |
| 3 | Engages with this story | Clear claim with reasoning | A new argument or angle |
| 4 | Engages with specific details of the story | Specific, well-reasoned | New angle with specific, substantive reasoning |

Why integer levels with defined descriptions, and not "give a score 0–1"? LLMs give
unstable, poorly calibrated floating-point scores. Anchored levels are far more
repeatable. We also set temperature 0, force structured JSON output, and cache results by
content hash, so the same submission always gets the same judgement.

Why show only the top-5 neighbours? It keeps the prompt small and focused. Semantic
retrieval already finds the items most likely to contain the same idea, so the judge
answers the narrow question "is this idea already *here*?".

### 2.4 Gates
`gate(level) = clip((level − 1) / 2, 0, 1)`, which gives levels 0,1,2,3,4 → **0, 0, 0.5, 1, 1**.

- **Relevance gate:** off-topic or passing mentions (R ≤ 1) get **exactly 0**, however novel
  they are. This is the "high novelty, low relevance is not rewarded" requirement. Generic
  four-day-week comments (R = 2) keep half their novelty.
- **Quality gate:** gibberish and one-liners get 0. This matters because **random text is
  the most "novel" input possible in embedding space**. Without this gate, word salad would
  score highest.
- **Multiplicative, not additive.** In an additive blend, very high novelty could make up
  for zero relevance. A product cannot do that.

### 2.5 Final score
`score = (0.4·N_emb + 0.6·N_llm) × gate(R) × gate(Q)`. It lies in [0, 1] by construction,
because every factor is in [0, 1].

The LLM gets more weight because it judges ideas. The embedding term keeps the score tied
to a signal that gives the same answer every run and catches near-duplicates that the
judge might be too generous about.

**Embedding-only fallback** (no LLM available): `score = N_emb × gate_emb`, where the
relevance gate is a cosine threshold to the fixed content, calibrated in Step 9. We
report it as a weaker baseline.

### 2.6 Behaviour over time
The ~50-item corpus is a fixed reference for evaluation. In production, each accepted
submission would be added to the corpus after scoring. That rewards the *first* person to
make a point, and later people who repeat it score lower. This is intended, but it makes
scores depend on arrival order (see Limitations).

### 2.7 Why not web or paper search
Novelty is measured **against peer submissions**, as the task requires, not against the
web. This is not plagiarism detection: 20 users making the same point in different words
are all *non-novel*, even though none of them copied anyone. Web search is left out of
the core pipeline for three reasons:
- a search would find the source article itself;
- results change over time, which would make the tests unreliable;
- it adds cost.

It stays a possible later extra.

## 3. Data

We build the dataset ourselves from a hand-written plan,
[data/angles.json](../data/angles.json). An LLM (Gemini, through the `LLM` interface)
writes the text, using [scripts/generate_data.py](../scripts/generate_data.py).

### 3.1 Why plan angles instead of asking for "50 comments"
If you ask an LLM for 50 comments, the ideas are random and nobody knows which ones are
novel. So the tests would have no ground truth. Instead, every generated submission is
assigned an **angle** (its core point), and the instructions say to make that point
"and no other major argument". Because we choose the angles, we know how common each
idea is.

### 3.2 Corpus: 50 existing submissions, deliberately skewed

| Tier | Angles | Items | Examples |
|------|--------|-------|----------|
| Popular | P1–P3 | 10 + 9 + 8 = 27 | reduces burnout, helps recruitment, slower service for residents |
| Medium | M1–M4 | 4 + 4 + 3 + 3 = 14 | overtime cost, 10-hour fatigue, childcare clash, "how is productivity measured?" |
| Rare | R1–R5 | 1 each = 5 | emissions, fairness to private-sector taxpayers, traffic, union lock-in, downtown lunch trade |
| Generic | G0 | 4 | "good idea" or "waste of money" with no argument |

The popular angles repeat points the article itself mentions (burnout, recruitment,
response times). That is what happens in real comment sections, and it is exactly what a
novelty scorer should not reward.

### 3.3 Golden set: 26 labeled new submissions

| Category | n | Built from | Expected score |
|----------|---|-----------|----------------|
| `novel_relevant` | 6 | **Held-back angles H1–H6**, which never appear in the corpus (e.g. 24/7 police/fire excluded → two-tier workforce; no control group in the pilot; January start confounds seasonal data) | 0.60–1.00 |
| `partly_novel` | 3 | A corpus angle plus one specific new idea | 0.30–0.75 |
| `paraphrase_common` | 5 | Fresh wording of P1, P2, P3, M1, M2 | 0.00–0.30 |
| `paraphrase_rare` | 3 | Fresh wording of R1, R2, R5 (already said once) | 0.00–0.50 |
| `offtopic_novel` | 4 | Thoughtful, original, off-topic. Includes a **hard negative**: a Riverton bike-lane proposal (same city and civic tone, different policy) | 0.00–0.10 |
| `low_effort` | 3 | Hand-written | 0.00–0.10 |
| `gibberish` | 2 | Hand-written word salad (passes the 10-word schema minimum) | 0.00–0.10 |

**Ground truth comes from how the data is built:** a held-back angle is novel because we
deliberately kept it out of the corpus. Nobody has to judge it after the fact. The
expected ranges map directly onto the success criteria in §5.

### 3.4 Making the text varied
Each submission gets its own **persona** (16 options, no repeats within an angle), a
**style** (8 options) and a **length** band (25–50, 50–110 or 110–180 words). Stance
rotates through the stances allowed for that angle. Generation runs at temperature 1.0,
with a fixed random seed (42) for these assignments. Without this, LLM-written text all
sounds alike, and embedding similarity would reflect writing style rather than ideas.

One API call per angle writes all of that angle's items together, with an instruction
not to reuse phrasing across them. That is 13 calls for the corpus and 21 for the golden
set. Every draft is validated against the `Submission` schema, and the call is retried
(up to 3 times) if validation fails.

### 3.5 Generation model
All 71 generated items come from **one model, `gemini-3.5-flash-lite`** (the `model` field on
each item). We first tried `gemini-3.8-flash`, but its free-tier quota ran out partway through.
We regenerated everything rather than mixing models: each model has its own writing
style, and embeddings would then partly measure *which model wrote the text* rather than
*which idea it contains*. A test (`test_single_generation_model`) enforces this.

### 3.6 Review and integrity checks
**Manual review:** all 21 generated golden items were read by hand. Every one makes its
assigned point. Things we noticed and **kept on purpose**, because they are realistic and
make the test harder:
- Some novel items also briefly repeat a common point. g005 (H5, move services online)
  opens with praise for staff well-being (P1), and g003 (H3) mentions burnout. The judge
  has to reward the *new* idea despite the familiar framing.
- Off-topic items sometimes show traits of their writer persona. The sci-fi review talks
  about a "municipal ledger" and "taxpayers", and the investing tip uses union language.
  They share words with the corpus but not ideas, so they are good relevance hard negatives.
- g018 (bike lanes) names Riverton and downtown businesses: same city and civic tone,
  different policy.

**Leakage check:** we scanned the corpus for distinctive terms from each held-back angle
(police/911, control group, disability, season/January, online/24-7, neighbouring towns).
There were two hits, both false positives ("chronic fatigue", "chronic vacancies"), so no
held-back idea appears in the corpus. The check now runs automatically in
[tests/test_data.py](../tests/test_data.py), together with plan counts, schema validity,
unique IDs and valid expected ranges.

## 4. Components
_To be written in Steps 4–7._ Covers the embeddings, relevance, novelty, and pipeline modules.

**Decision: provider-agnostic model access.** The scoring logic never calls a vendor SDK
directly. It depends on two small interfaces:

- `Embedder.embed(texts) -> vectors`: used for scoring
- `LLM.generate(prompt) -> text`: used for the rubric judge (§2.3) and for generating synthetic data

Gemini (`google-genai`) is the default adapter because that key is available. The default model is `gemini-3.8-flash` (`GEMINI_MODEL`). When it returns a transient error (429/5xx), the adapter tries `gemini-3.7-flash`, then `gemini-3.5-flash` (`GEMINI_FALLBACK_MODELS`). A 429 is handled according to its quota type. A **per-minute** limit waits for the `retryDelay` the API returns. A **per-day** limit removes that model from the chain, because retrying only uses up more quota. If every model is busy (5xx), it waits (5 → 80 s, exponential backoff) and tries the whole chain again. `CachedLLM` wraps any provider with an on-disk cache (`.cache/llm/`, keyed by hash of prompt, schema and temperature). This makes reruns free and repeatable and lets an interrupted run resume. Each generated data item records which `model` wrote it. Other
providers can be added as extra adapters and selected with a `PROVIDER` environment
variable. Tests use a deterministic fake embedder, so they run offline without any API key.

### 4.1 Embeddings ([novelty/embeddings.py](../novelty/embeddings.py))

| Class | Role |
|-------|------|
| `Embedder` (protocol) | `embed(texts) -> (n, d)` L2-normalized float32, so cosine = dot product |
| `GeminiEmbedder` | `gemini-embedding-2` (`EMBED_MODEL`), `task_type=SEMANTIC_SIMILARITY`, batches of 50, per-minute 429 → wait `retryDelay`, 5xx → backoff |
| `CachedEmbedder` | per-text `.npy` cache in `.cache/emb/`, keyed by (embedder id, text); only uncached texts reach the API |
| `FakeEmbedder` | offline, deterministic feature hashing of words and bigrams. Used by unit tests. It measures word overlap, **not meaning**, and is never used for evaluation |

`EMBED_PROVIDER=gemini|fake` selects the provider.

**Choosing the model.** We tested both available models on a burnout paraphrase and an
off-topic sourdough text:

| Model | sim(paraphrase) | sim(off-topic) | gap |
|-------|-----------------|----------------|-----|
| `gemini-embedding-2` | 0.845 | 0.553 | **0.29** |
| `gemini-embedding-001` | 0.892 | 0.723 | 0.17 |

`gemini-embedding-2` separates ideas better, so we chose it.

**Gotcha found in testing.** `gemini-embedding-2` is multimodal. Given a *list of strings*,
it returns **one** embedding for all of them combined. Each text must be wrapped in its own
`types.Content`. The adapter does this and also checks that the number of returned vectors
matches the number of inputs.

**Diagnostics on the real data** ([scripts/inspect_embeddings.py](../scripts/inspect_embeddings.py)):

| Check | Result | Meaning |
|-------|--------|---------|
| Corpus within-angle vs between-angle mean similarity | 0.821 vs 0.741 (gap 0.08) | All submissions share a topic, so similarities are squeezed into a narrow band. This is why §2.2 normalizes by percentile rather than using raw cosine |
| Nearest-neighbour shares the item's angle | **87%** | Embeddings capture ideas reasonably well, even with persona and style noise |

Preview of the golden set. `N_emb` here is the percentile-normalized embedding novelty
from §2.2, with no gates:

| Category | sim → fixed content | max sim → corpus | N_emb mean [min–max] |
|----------|--------------------|------------------|----------------------|
| novel_relevant | 0.737 | 0.818 | **0.90** [0.78–0.98] |
| partly_novel | 0.746 | 0.857 | 0.58 [0.46–0.78] |
| paraphrase_common | 0.739 | 0.865 | 0.52 [**0.10–0.76**] |
| paraphrase_rare | 0.722 | 0.860 | 0.73 [0.56–0.84] |
| offtopic_novel | 0.610 | 0.668 | **1.00** |
| low_effort | 0.614 | 0.668 | **1.00** |
| gibberish | 0.522 | 0.584 | **1.00** |

**What this tells us about the design:**
1. **Novel ideas stand out.** Every held-back-angle item has N_emb ≥ 0.78, and the mean
   ordering is novel > partly novel > common paraphrase, as intended.
2. **Embeddings alone don't reliably catch paraphrases.** One common-angle paraphrase
   reaches N_emb 0.76. Persona and style differences (a nurse's anecdote vs a student's
   rant) push same-idea texts apart. This is why §2.3 adds the LLM novelty judgement.
3. **Junk is the most "novel" input possible.** Off-topic, low-effort and gibberish items
   all get N_emb = 1.00. This confirms that the multiplicative gates in §2.4 are
   essential and not optional.
4. **Embedding relevance breaks on the hard negative.** The off-topic bike-lane item
   (g018) has similarity **0.681** to the fixed content. The least similar truly relevant
   golden item has **0.683**. A margin of 0.002 is not a usable threshold, so relevance
   must come from the LLM rubric. In the embedding-only fallback, SC2 is expected to fail
   on g018, and we will report it that way.

A possible improvement if paraphrases stay a problem: have the LLM rewrite each
submission as a one-sentence core claim, and embed that claim instead of the raw text.
This strips persona and style before comparison.

## 5. Success criteria
These are defined **before** building the data or the model code, so they can't be
adjusted to fit the results later. They are measured on the golden set in Step 9.
Categories come from §3.

| ID | Criterion | Target |
|----|-----------|--------|
| SC1 | Every score lies in [0, 1] | 100% (also unit-tested) |
| SC2 | **Novel but off-topic** items score ≤ 0.10 | 100% |
| SC3 | Low-effort and gibberish items score ≤ 0.10 | 100% |
| SC4 | Paraphrases of *common* corpus takes score ≤ 0.30 | ≥ 90% |
| SC5 | Novel and relevant items (held-back angles) score ≥ 0.60 | ≥ 90% |
| SC6 | Ranking: P(score(novel) > score(paraphrase)) across all pairs | ≥ 0.95 |
| SC7 | Median ordering: novel > partly novel > paraphrase | holds |
| SC8 | Repeatability: re-scoring with the cache disabled changes the score by ≤ 0.05 | ≥ 90% of a sample |

The embedding-only baseline is reported against the same criteria for comparison, but it
is not required to pass them.

## 6. Limitations and future work
_To be written in Step 10._
