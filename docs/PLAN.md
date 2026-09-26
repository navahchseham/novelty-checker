# Build Plan

The first iteration (Riverton, Steps 0–4) is archived with its own plan in
[../archive/riverton/docs/PLAN.md](../archive/riverton/docs/PLAN.md). This plan covers the
rebuild on the design spec ([design_spec.pdf](design_spec.pdf)).

## Requirements (Problem Statement 3)

| Requirement | Where |
|-------------|-------|
| Content shape: 3 user fields, fixed content ≤ 100 words | DESIGN §1 |
| Reward novelty relative to other submissions, keep a minimum relevance | DESIGN §2–3 |
| Evaluate against ≈ 50 other submissions (LLM-generated synthetic data) | DESIGN §4, `data/` |
| Automated tests: novel rewarded, non-novel not, **high novelty + low relevance not rewarded** | `tests/`, EVALUATION |
| Score normalized to [0, 1] | DESIGN §2 |
| Write-up: design, rationale, success criteria, level of achievement | DESIGN, EVALUATION |
| Coding-agent disclosure | AGENT_COLLABORATION |

## Steps

| Step | What | Output | Status |
|------|------|--------|--------|
| R0 | Archive the Riverton iteration; adopt the spec | `archive/riverton/`, `docs/design_spec.pdf` | ✅ |
| R1 | Schema, fixed content, parameters | `novelty/schema.py`, `data/fixed_content.json`, `novelty/config.py` | ✅ |
| R2 | Data plan and generation (50 corpus + 23 golden) | `data/angles.json`, `scripts/generate_data.py`, `data/corpus.json`, `data/golden.json` | ✅ |
| R3 | Relevance floor and duplicate gate | `novelty/pipeline.py` | ✅ |
| R4 | Point extraction (+ LLM relevance) and canonical pool | `novelty/extract.py`, `novelty/pool.py` | ✅ |
| R5 | Judge and rarity scoring | `novelty/judge.py`, `novelty/scoring.py` | ✅ |
| R6 | Pipeline and CLI | `novelty/pipeline.py`, `scripts/score.py` | ✅ |
| R7 | Tests: offline (stubbed LLM) and live | `tests/` | ✅ |
| R8 | Calibrate, build pool, evaluate, write up | `scripts/build_pool.py`, `scripts/evaluate.py`, `docs/EVALUATION.md` | ✅ |
| R9 | Entry point, web UI and visualizations | `main.py`, `app.py`, `docs/VISUALIZATIONS.md`, `docs/screenshots/` | ✅ |

## Progress log

- **2026-09-26: Rebuild on the design spec.** Reviewed the spec and proposed changes 1–6
  (DESIGN §3), backed by the Riverton measurements. Archived the Riverton work. Built R1–R7.
- **Data.** 50 corpus items and 19 generated golden items plus 4 hand-written ones, all
  from `gemini-3.5-flash-lite`. Read by hand. The leakage scan's two hits were false
  positives ("camera gear", a factual mention of the van exemption), and the probes were
  narrowed.
- **Finding: article restatements dilute novel items.** Generated novel items often open by
  restating the article. Change 3 was refined: article-matching points are neutral
  (excluded from `N_point`).
- **Finding: the offline fake embedder can't test `N_sem` thresholds.** It measures word
  overlap, not meaning. Unit tests now check logic (m, `N_point`, ordering), and absolute
  thresholds moved to live tests.
- **First pool build.** Two problems: the duplicate gate rejected 5 distinct corpus comments
  (cosine 0.909–0.937), and over-split extraction spread one idea across several canonical
  points, so repeated takes stayed at m ≈ 1 and scored 0.5–0.7. Fixes: calibrate `dup_max`
  above every corpus pair; extraction merges statements supporting one argument; the judge
  lists all synonym points, and m is the number of distinct submissions containing any of
  them (DESIGN §3, changes 8–9). 135 offline tests pass. Rebuilding.
- **First full evaluation: 9/12.** SC3 failed because a bare "great idea" was extracted as
  a new point. SC7 failed because of the spec's λ contradiction plus `N_sem` ignoring a
  single close match. SC2 failed on g011, where the generator's taxi-driver persona added a
  genuinely new claim; we left the label as it was. Fixes: extraction skips bare evaluations
  and article restatements (change 11); `N_sem` includes the closest match (change 10).
- **Incident: the rebuild crashed at item 20/50** on `Connection reset by peer`. The retry
  code handled API errors but not network errors. The evaluation that followed had silently
  used the *previous* pool, so its results were discarded. Fixes: both adapters retry
  `httpx.TransportError`, and the build→evaluate chain now uses `pipefail`, so an
  evaluation can never run on a stale pool.
- **Clean rebuild and re-evaluation: 10/12.** The first clean run (after the crash fix)
  was 9/12. The judge was biased toward "new": my prompt said an added consequence is not
  a match, and retrieval sometimes hid an existing match. Fixed with a looser matching rule,
  and the judge now sees every point while the registry has ≤ 150 points (change 12). The
  pool shrank to 39 canonical points and repeated takes score 0.03. Both remaining failures
  (SC2 on g011, SC7 first mover) score **0.60** for the same reason: m = 1, and λ = 2 keeps
  w(1) = 0.61. An offline sweep (`scripts/lambda_sweep.py`): λ = 0.4 passes all 12 on the
  dev set; λ = 0.5 lands exactly on 0.20. **This is a decision for the developer**; see the λ trade-off in EVALUATION.
- **Golden set is now a development set** (we iterated on its failures). A fresh
  **holdout** plan was added (`angles.json` → `holdout`: new angles H7–H9, new off-topic
  topics, new hand-written items, seed 1234), to be generated and evaluated once, after the
  design is frozen. The generalized generator reproduces the dev set exactly (29/29 cached
  prompts).
- **Daily quota for `gemini-3.5-flash-lite` exhausted** during the extractor-consistency
  check. Evaluation now saves results before that check, and the check is opt-in
  (`--consistency`). The dev evaluation reran entirely from cache.
- **Groq dev evaluation: 11/12.** λ = 0.4 and the matching fixes carried over to a different
  judge model: the first-mover paraphrase drops from 1.00 to 0.15. The only failure is SC2
  on g011. Both judge models independently found that its "cab drivers earn more fares" is a
  new claim, so it is **label noise**; the label was kept, not changed.
- **Design frozen** (`sha256(novelty/*.py + params.json + pool.json)` = `07ef10d009cfe2c0`).
  The holdout set is evaluated **once**, on this version, and its result is reported as-is.
- **Holdout (run once): 9/12.** All novel items ≥ 0.99; all irrelevant items exactly 0.
  The failures (SC2, SC3, SC7) come from one mechanism: one sub-point counted as new earns
  1/n credit in a mostly repeated submission. Contributing causes are persona-specific
  consequences and negatively phrased bare evaluations. Documented, not patched (EVALUATION
  §3, §5). Uncached extraction is not repeatable (2/3 runs differed), so that live test is a
  documented `xfail`.
- **Packaging.** Added the single entry point `main.py` and a Streamlit UI (`app.py`),
  tested headlessly: a novel idea scores 1.00, its paraphrase 0.21, off-topic 0.00.
- **Visualizations.** The developer specified five views (`docs/VISUALIZATIONS.md`) and
  implemented them from real pipeline output: gate checklist, score-component bars,
  point-level LLM view, relevance-vs-novelty scatter, novelty distribution, and a
  leaderboard, with the submitter's comment always marked "You". README screenshots were
  captured from the running app; the point-level chart's overlapping rows were fixed.
- **Docs and UI explanations.** Added the topic rationale and a full explanation of stances
  (each stance's meaning, and that stance is judge context only, never a score factor) to
  DESIGN §1 and the README. In the app, each example button explains on hover (and above
  the form once loaded) what it demonstrates and its expected score, and each stance option
  shows its meaning as a caption.
