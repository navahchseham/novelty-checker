# Novelty Scoring Visualizations


> **Status: implemented** in the Streamlit app ([app.py](../app.py)): the gate checklist,
> score-component bars and point-level LLM view on the **Score a submission** tab, and the
> scatter, distribution and leaderboard on the **Visualizations** tab (screenshots in the
> [README](../README.md#visualizations)). The app computes every view from **real pipeline
> output** (`data/corpus_scores.json` for the pool, live results for new submissions). The
> numbers in *this* document are the original illustrative values and use the original
> design's parameters (α = 0.7, λ = 2, `DUP_MAX` 0.9); the implemented values are in
> [DESIGN.md §5](DESIGN.md#5-parameters-and-calibration).

## Purpose

Five views explain the novelty score: two trace a single submission, three show the whole pool.

| View | Question it answers | Audience |
| --- | --- | --- |
| Score breakdown | Why did this submission get this score? | Developer, debugging |
| Point-level LLM view | What did the LLM judge decide about each point? | Developer, debugging |
| Relevance vs novelty scatter | Do the gates block novel but irrelevant posts? | Assessor |
| Novelty distribution | Where does a new comment fall against the pool? | Both |
| Leaderboard | Which comments are most novel, and who was first with an idea? | Both |

**All numbers here are illustrative.** They are representative values built from the corpus themes and test cases, not pipeline output. The layouts and calculations are final; the numbers are replaced once the pipeline runs.

**You are here.** Every view marks the current submitter's own comment, labelled "You", so the graphs read as personal feedback rather than a general overview. The other submissions stay visible as faint context. When the submitter's comment was blocked by a gate and scored 0.0, its marker shows the reason rather than a rank. Here the displacement comment plays the role of "You".

## Score breakdown

The displacement submission scores 0.73: three of its four points are new or nearly new, and every gate passes.

> **The charge will just move the jam to our street** · stance: support\_with\_changes Downtown traffic will fall, but drivers will cut through and park on the streets just outside the zone. The council should monitor edge streets during the consultation.

**Step 1: Gates.** All three pass, so G = 1.

| Gate | Value | Threshold | Result |
| --- | --- | --- | --- |
| Validate | 3 of 3 fields valid | All fields | Pass |
| Relevance | 0.58 (headline 0.41, body 0.65) | 0.30, body 0.35 | Pass |
| Duplicate | Highest similarity 0.47 | Below 0.90 | Pass |

**Step 2: Atomic points.** The judge matches each point against the neighbours' cached points.

| Atomic point | Matches existing point | Prior mentions m | Rarity weight w(m) |
| --- | --- | --- | --- |
| Traffic shifts to streets outside the zone | None | 0 (first) | 1.00 |
| Council should monitor edge streets | None | 0 (first) | 1.00 |
| Drivers park in residential edge streets | "weekend and edge parking" | 1 | 0.61 |
| Downtown traffic will fall | "charge cuts traffic" | 11 | 0.00 |

N\_point = (1.00 + 1.00 + 0.61 + 0.00) / 4 = 0.65

**Step 3: Combine.**

| Component | Value | Weight | Contribution |
| --- | --- | --- | --- |
| Point novelty N\_point | 0.65 | α = 0.7 | 0.46 |
| Semantic novelty N\_sem (91st percentile) | 0.91 | 1 − α = 0.3 | 0.27 |
| Quality factor Q (quality 0.90) | 1.00 | multiplier | × 1.00 |
| **Final score S** |  |  | **0.73** |

**The same breakdown for the other two outcomes:**

| Stage | Low-income paraphrase | Biryani recipe |
| --- | --- | --- |
| Relevance | 0.62, pass | 0.03, **fail** |
| Duplicate | 0.71, pass | not reached |
| Atomic points | 2 points, m = 10 and 7 | not reached |
| N\_point | 0.02 | not reached |
| N\_sem | 0.12 | 0.99 (shown for reference) |
| **Final score S** | **0.05** | **0.0** |

The biryani recipe is the most semantically novel text in the set, yet it stops at the relevance gate. That column is the required outcome, visible in one row.

In the app this renders as a card per submission: gate checkmarks at the top, the points table with first-mover badges, and the component bars summing to the final score.

## Point-level LLM view

This view exposes the LLM judge's verdict, which the other charts compress into a single axis. For one submission, it lists each atomic point, whether the judge matched it to an existing point, how many pool submissions already made it, and the rarity weight that follows.

&#91;embedded content: illustrative values · 4 atomic points, 1 submission\]

The two new points carry full weight; the widely-made "traffic will fall" point carries none. This is the direct evidence that the idea-novelty step works: it rewards a genuinely new claim even when the wording resembles existing comments, and it ignores a familiar claim however it is phrased.

In the app, hovering a matched point shows the existing point it was tied to, so a reviewer can check the judge's call. When a match looks wrong, that is the row to correct, and it is where a prompt or model change would show up first.

## Relevance vs novelty

Every off-topic test case sits in the top novelty band and still scores 0.0, because it falls left of the relevance gate.

&#91;embedded content: illustrative values · 50 corpus items + 10 test cases\]

The ISP "network congestion charge" post clears the embedding gate but scores 0.0: the LLM relevance check caught it. The exact copy and the paraphrases sit at the bottom, far below the rewarded band. About one in five corpus items scores 0.6 or more, which matches the calibration target.

## Novelty distribution

A new comment's novelty percentile is the share of the pool's leave-one-out scores below its own. The displacement comment beats 91% of the pool.

**Leave-one-out.** Score each of the 50 pooled submissions as if it were the new arrival: remove it, compare it with the other 49, and record its raw novelty. Those numbers, sorted, form the reference distribution. Build it once; do not recompute it per submission.

**Scoring a new comment is cheap.** Compute its raw novelty once, then binary-search its place in the sorted list to get the percentile. No comparison against the full pool is repeated.

**Updating the pool is incremental.** When the new comment joins, it changes the raw novelty only of submissions for which it is now one of the five nearest neighbours, a subset of the neighbours the retrieval step already found. Recompute those few scores and insert the new one into the sorted list. This is close to constant work per submission, not the order n-squared of a full rebuild.

**Refresh on a schedule.** Incremental updates drift slightly over time, so a full recompute nightly, or every few thousand submissions, keeps the distribution honest without ever blocking a live submission.

&#91;embedded content: illustrative values · 50 pool submissions, 3 new comments\]

The recipe is further right than anything in the pool, at the 99th percentile, yet it scores 0.0 at the relevance gate. The distribution measures difference only; the gates decide whether difference counts.

## Leaderboard

Your submission, the displacement comment, ranks 7th of 55 eligible submissions: 88th percentile, with two first-mover points. Your row is pinned even when it falls outside the top few.

| Rank | Submission | Source | Score S | Percentile | New points | Rarest point (share of pool) | First mover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Weekends are busy too | Pool | 0.98 | 98th | 2 of 2 | 0% | First ×2 |
| 2 | Pedestrians spend more than drivers | Pool | 0.93 | 96th | 2 of 3 | 0% | First ×2 |
| 3 | Why do residents get a discount? | Pool | 0.89 | 94th | 1 of 2 | 0% | First ×1 |
| 4 | Ring-fence the revenue | Pool | 0.84 | 92nd | 2 of 3 | 2% | First ×1 |
| 5 | Safer streets for cyclists | Pool | 0.80 | 90th | 1 of 2 | 0% | First ×1 |
| 6 | Need to see the consultation | Pool | 0.76 | 88th | 1 of 2 | 2% | None |
| 7 | **You (displacement)** | New | **0.73** | **88th** | 3 of 4 | 0% | First ×2 |
| 8 | **Camera privacy** | New | **0.71** | **86th** | 2 of 3 | 0% | First ×2 |
| 9 | **Dynamic pricing** | New | **0.68** | **84th** | 2 of 3 | 0% | First ×1 |

**How each column is computed**

- **Score S:** the final 0.0–1.0 score from the pipeline.
- **Percentile:** the share of the pool's leave-one-out scores below S.
- **Rarest point:** m / N, where m is the number of pool submissions already making that point and N is the pool size. 2% means one of 50.
- **First mover:** the number of the submission's points with m = 0 at the moment it was scored. The badge is frozen then, so later copies do not remove it.

First mover is a badge, not a percentage. It is the extreme end of rarity (m = 0), so a separate percentile would repeat the rarity column with false precision.

**Not ranked, blocked by a gate:** recipe, football report, copied headline and ISP charge (relevance); exact copy (duplicate). **Bottom of the ranking:** the low-income paraphrase at 0.05 and generic support at 0.02, both with no new points.

## Implementation

All four views read one stored record per submission. The pipeline writes that record once at scoring time, so the views never call an LLM.

```json
{
  "id": "s-0187",
  "scored_at": "2026-09-26T10:14:00Z",
  "headline": "...", "body": "...", "stance": "support_with_changes",
  "gates": {"valid": true, "relevance": 0.58, "body_relevance": 0.65,
            "llm_relevant": null, "max_similarity": 0.47, "passed": true},
  "points": [
    {"text": "traffic shifts to streets outside the zone",
     "matches": null, "prior_mentions": 0, "weight": 1.0}
  ],
  "n_point": 0.65, "n_sem_raw": 0.51, "n_sem": 0.91,
  "quality": 0.9, "score": 0.73,
  "first_mover_count": 2
}
```

| View | Built from | Recomputed when |
| --- | --- | --- |
| Score breakdown | One record: gates, points, components | Never; frozen at scoring |
| Relevance vs novelty scatter | Every record: relevance, n\_sem, score | A submission is added |
| Novelty distribution | Leave-one-out n\_sem\_raw for the pool, plus the new record | Pool changes (batched for large pools) |
| Leaderboard | Every passing record, sorted by score; percentile against the pool | A submission is added |

**Build steps**

1. Run the pipeline over the 50-item corpus and the 10 test cases, writing one record each.
2. Run leave-one-out over the corpus to produce the reference distribution.
3. Render the views from the records. Replace the illustrative numbers in this doc with the output.
4. Add the views to the test report, so a failing test links to its breakdown card.

The views are also a check on the pipeline: if a test case lands in the wrong region of the scatter, the breakdown card shows which stage put it there.
