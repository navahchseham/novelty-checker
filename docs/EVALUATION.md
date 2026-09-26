# Evaluation: Level of Achievement

**Bottom line.** The hard requirement holds on data never seen during development:
**novel and relevant ideas are rewarded, and off-topic content scores exactly 0 even
when it is maximally "novel"**. Copies, gibberish and prompt injection are stopped. The
weak point is *non-novel* content that picks up one sub-point the judge counts as new.
That drove all three holdout failures, and the next steps in §5 target it.

| Set | Used for | Criteria passed |
|-----|----------|-----------------|
| Development (23 items) | iterating on the design | **11 / 12** |
| **Holdout (19 items)** | scored **once**, after freezing the design | **9 / 12** |

Reproduce: `python -m scripts.evaluate --set dev` / `--set holdout`. Raw results are in
`data/eval_results.json` and `data/eval_holdout.json`. Live tests: `pytest -m live`.

## 1. Method

- **Pool:** the 50-item corpus was scored through the full pipeline in shuffled order and
  added as it went, exactly like live submissions. The result is 49 submissions and 46
  canonical points (one corpus item had no claims of its own; see §4).
- **Golden items** are scored against that pool **without** being added, so each is judged
  independently. The first-mover test adds the novel item to a *copy* of the pool, then
  scores its paraphrase.
- **Why two sets.** We fixed the design by looking at the development set's failures, so
  its score is optimistic. The **holdout** was planned in advance and written by a different
  model (`qwen/qwen3.8-27b`), with three **new** novel angles (carpool exemption, tradespeople
  passing on the charge, ride-hail per-trip pricing), new off-topic topics and new
  hand-written items. It was evaluated once, on a frozen design (fingerprint
  `07ef10d009cfe2c0` in PLAN.md), and nothing was changed afterwards.
- **Models:** extraction and judge `openai/gpt-oss-120b` (Groq), embeddings
  `gemini-embedding-2`, corpus and dev writer `gemini-3.5-flash-lite`, holdout writer
  `qwen/qwen3.8-27b`.

## 2. Results by criterion

| ID | Criterion | Dev | Holdout |
|----|-----------|-----|---------|
| SC1 | Novel and relevant items score > 0.6 | ✅ 0.72–1.00 (6/6) | ✅ 0.99, 0.99, 0.99 (3/3) |
| SC2 | Paraphrases of common takes score < 0.2 | ❌ 3/4 (g011: 0.95) | ❌ 2/4 (h007 0.41, h008 0.98) |
| SC3 | Generic comment scores < 0.2 | ✅ 0.09 | ❌ 0.34 |
| SC4 | Irrelevant items score **exactly 0.0** | ✅ 4/4 | ✅ 4/4 |
| SC5 | …and they had raw `N_sem` ≥ 0.9 (blocked by relevance, not luck) | ✅ all 1.00 | ✅ all 1.00 |
| SC6 | Exact copy and stance-only flip score 0.0 | ✅ | ✅ |
| SC7 | First mover > 0.6; its paraphrase afterwards < 0.2 | ✅ 0.99 → 0.15 | ❌ 0.99 → 0.37 |
| SC8 | Point made once > point made 5× | ✅ 0.59 vs 0.10 | ✅ 0.45 vs 0.11 |
| SC9 | Median novel > partly novel > paraphrase | ✅ 0.99 > 0.30 > 0.08 | ✅ 0.99 > 0.55 > 0.27 |
| SC10 | Fluent gibberish scores 0.0 | ✅ | ✅ |
| SC11 | Prompt injection scores < 0.2 | ✅ 0.00 | ✅ 0.10 |
| SC12 | Every score in [0, 1] | ✅ | ✅ |

**The problem statement's three requirements:**

| Requirement | Evidence |
|-------------|----------|
| Truly novel content is rewarded | SC1: 9/9 novel items across both sets score ≥ 0.72; 8 of them ≥ 0.98 |
| Non-novel content is not rewarded | Mostly holds: median paraphrase 0.08 (dev) / 0.27 (holdout); rarity and first-mover decay work (SC7 dev, SC8). **Not guaranteed**: see §3 |
| High novelty but low relevance is not rewarded | **Holds without exception**: 8/8 off-topic items across both sets score exactly 0.0, every one with `N_sem` = 1.00 |

## 3. Why the failures happened

All failures share one mechanism. **`N_point` is the mean over a submission's extracted
points, so a single sub-point the judge counts as new earns 1/n credit.** When a
submission that is basically a repeat picks up such a sub-point, it escapes the < 0.2 bar.

| Item | Score | Sub-point counted as new | Assessment |
|------|-------|--------------------------|------------|
| g011 (dev, "traffic relief") | 0.95 | "It will increase cab drivers' earnings" | The taxi-driver persona added a **genuinely new consequence**. Both judge models (Gemini and gpt-oss) found it independently. The **label** is wrong, and we left it unchanged |
| h008 (holdout, "traffic relief") | 0.98 | "Less congestion will allow faster deliveries"; also an article restatement (buses and bike lanes) not detected | The same **persona-consequence** pattern as g011, now seen twice, so it is systematic. Is "faster deliveries" novel? It is a new *consequence*, but a predictable one |
| h016 (holdout, generic) | 0.34 | "The council habitually makes poor decisions" | **Extraction failure.** "A bare evaluation is not a point" worked for positive phrasing (dev) but not for **negative** phrasing, which the dev set never tested |
| h007 (holdout, "distrust revenue") | 0.41 | "Projected traffic reduction lacks independent verification" | Arguably a real new claim added by the writer; borderline |
| SC7 holdout (h012 after h003) | 0.37 | "Exempting taxis from the charge is unfair" | The first mover's point was "the taxi exemption is useless because of idle Ubers", and the judge did not treat these as the same claim. Its other points did match (m = 1) |

## 4. Other findings

- **Extraction is not repeatable.** Uncached re-extraction at temperature 0 with
  `gpt-oss-120b` gave a different number of points on 2 of 3 submissions (best-match
  similarity down to 0.66). The pipeline extracts **once** and stores the points, as the
  spec requires, so stored scores do not drift. Re-scoring from scratch, however, would
  not reproduce them exactly. The live test is kept as a documented `xfail`.
- **Empty-of-ideas comments score 0.** Corpus item c005 is pure enthusiasm plus a
  restatement of the article's expected effect. It had no points of its own, scored 0, and
  was not pooled.
- **The first mover is rewarded and repetition decays**, both when building the corpus pool
  (later same-theme items score 0.02–0.15) and in SC7 on dev (1.00 → 0.15).
- **Changing the judge model worked as intended.** The switch from Gemini to Groq (after the
  Gemini quota ran out) required only a new adapter. Dev results went from 10/12 (Gemini,
  λ = 2) to 11/12 (Groq, λ = 0.4).
- **Calibration** (corpus only; see DESIGN §5): relevance floors 0.584 / 0.601 (lowest corpus
  values − 0.05); neighbour threshold t = 0.781 (80th percentile of pairwise similarity,
  about 10 neighbours each); duplicate cut-off 0.957 (highest corpus pair 0.937 + 0.02).

## 5. What we would do next

Each of these targets the failure mechanism in §3. Any fix must be checked on a **new**
holdout, not on the one reported here.

1. **Weight points by rarity using max-plus-mean instead of the mean**, or ask the judge to
   mark each point **central or peripheral** and weight peripheral points down. A
   submission's novelty should rest on its main argument, not on a stray sub-point.
2. **Persona-specific consequences:** extend the matching rule so that "the same effect,
   felt by a particular group" counts as the existing claim unless it introduces a
   different mechanism.
3. **Negative bare evaluations:** add "bad idea / typical council / won't work" examples to
   the extraction rule, and add such items to the dev set.
4. **Repeatable extraction:** use self-consistency (extract 3 times and keep points found
   in at least 2 runs), or a model with deterministic decoding.
5. **Bigger test sets.** With 19–23 items, one item moves a criterion. Per-category rates
   over about 100 items would give tighter estimates.
