# Coding Agent Disclosure

The submission rules require disclosure of coding-agent use and an explanation of how the
agent was directed. This file is that record.

**Agent used:** Claude Code (Anthropic), in the VS Code extension.


**Developer:** individual entry; all direction, review and decisions are the developer's.

## Working method

Every piece of work followed the same developer-led loop:

```
1. Developer divides the work     ─► defines the subtask to build next, in small steps
2. Agent follows through          ─► investigates, then proposes an approach and suggestions
3. Developer reviews              ─► accepts, rejects or adds feedback to the suggestions
4. Implement                      ─► the agent builds the reviewed version, with tests
5. Verify and document            ─► tests run, results reported, docs updated, then the next subtask
```

The developer set the structure of the work: first by directing a step-by-step
breakdown of the problem statement, and then with their own design document
([design_spec.pdf](design_spec.pdf)), whose pipeline stages became the build steps.
Nothing moved to the next subtask without the developer's go-ahead. Design decisions and
their evidence are in [DESIGN.md](DESIGN.md), and the step history is in [PLAN.md](PLAN.md).

## Phase 1: Problem breakdown and first prototype

| # | Developer direction | Agent suggestion | Developer review / feedback | Implemented |
|---|---------------------|------------------|-----------------------------|-------------|
| 1 | Chose Problem Statement 3. Directed: divide it into small subtasks, build one at a time, document each | Mapped every PS3 requirement to a numbered subtask; proposed an 11-step plan | Approved the plan; moved the project into the GitHub repo folder `hack/` | Project scaffold, docs skeleton, `.gitignore` |
| 2 | Any LLM provider may be used, not just Gemini | Put all model calls behind provider-agnostic `LLM` / `Embedder` interfaces | Approved | Interfaces with a Gemini adapter, plus an offline fake for tests |
| 3 | Proposed the approach: semantic search across a database, plus a **rubric-based LLM-as-a-judge** | Kept both of the developer's core ideas. Suggested that novelty be measured against *peer submissions* (as the task defines it) rather than web or paper search, which would find the source article itself | Agreed; web search kept as an optional extension | Semantic search over the submission pool + LLM rubric judge |
| 4 | Subtask: content shape (3 user fields, fixed content ≤ 100 words) | Suggested a news article with headline / body / stance, keeping stance out of the embedded text | Approved | Schema and tests |
| 5 | Asked how the dataset should be built | Suggested an *angle plan*: deliberately clustered themes, plus **held-back angles** used only for "novel" test items, so ground truth is known by construction | Approved; reviewed the angle plan before any generation | Generator, corpus of 50, labeled test set, leakage checks |
| 6 | Subtasks: scoring design, then embeddings | Proposed success criteria before building; measured embedding behaviour on the real data | Reviewed the measurements | Scoring math, embedding layer, diagnostics |

## Phase 2: The developer's design

| # | Developer direction | Agent suggestion | Developer review / feedback | Implemented |
|---|---------------------|------------------|-----------------------------|-------------|
| 7 | Wrote a full **design document**: gates first, atomic-point extraction, LLM judge for point matching, rarity decay e^(−m/λ), embedding cross-check, and a test plan. Asked for a review | Recommended adopting it, with 6 refinements, each backed by a measurement from Phase 1 (e.g. cosine relevance alone could not separate a hard negative: 0.681 vs 0.683) | Accepted the refinements; directed a rebuild on the design, with the first prototype archived | [DESIGN.md](DESIGN.md) §3, changes 1–6 |
| 8 | Directed implementing each step of the design | Built the stages in order: schema → data → gates → extraction → pool → judge → scoring → pipeline → tests | Reviewed progress at each stage | 139 offline tests + live tests |
| 9 | Directed running the pipeline on the full corpus | Inspected results item by item and suggested fixes for what the data exposed: a duplicate threshold too tight for the embedding model, and points split too finely to count repeats | Approved the fixes | Changes 8–9 |
| 10 | Directed evaluation against the success criteria | Suggested fixes for the failures, and suggested a separate **holdout test set**, evaluated once after freezing the design, so reported results are not tuned to the test data | Approved the holdout methodology | Changes 10–12; holdout plan with new, unseen angles |
| 11 | **Decided the rarity decay: λ = 0.4**, after reviewing an offline comparison of λ = 2, 1, 0.7, 0.5 and 0.4 | Presented the trade-off: the design's own first-mover test needs a steep decay | Chose λ = 0.4 | Change 13 |
| 12 | **Decided to switch the judge to Groq** and provided the API key | Suggested `gpt-oss-120b` as the judge and a *different* model (`qwen`) to write the holdout, so the judge never scores its own model's text; key stored only in the git-ignored `.env` | Approved | `GroqLLM` adapter; change 14 |
| 13 | Directed the final evaluation | Froze the design (fingerprint recorded), ran the holdout once, and reported the results as-is with a failure analysis and next steps | Reviewed the results | [EVALUATION.md](EVALUATION.md): dev 11/12, holdout 9/12 |

## Phase 3: Packaging

| # | Developer direction | Agent suggestion | Developer review / feedback | Implemented |
|---|---------------------|------------------|-----------------------------|-------------|
| 14 | Update `.gitignore` for the project | Keep `data/` tracked (the submission needs the dataset); add `.env.example` | Approved | `.gitignore`, `.env.example`, secret scan |
| 15 | Add a single `main.py` entry point | One command per script, so each command has a single implementation | Approved | `main.py` with `score`, `evaluate`, `build-pool`, `generate`, `lambda-sweep` |
| 16 | Build a Streamlit UI | Per-session pool, so reviewers can watch the first-mover effect without changing the saved data | Approved | `app.py`, tested headlessly (novel 1.00 → its paraphrase 0.21 → off-topic 0.00) |
| 17 | **Specified five visualization views** ([VISUALIZATIONS.md](VISUALIZATIONS.md)): score breakdown, point-level LLM view, relevance vs novelty, novelty distribution, leaderboard, each marking the submitter as "You" | Implemented in a separate session (Claude Sonnet 5) from real pipeline output rather than the doc's illustrative numbers | Reviewed and committed | Visualizations tab, gate checklist, component bars, point-level view (`b2aef5c`) |
| 18 | Asked to document the new views and explain what each stance does | Verified the app headlessly; moved the visualization doc into `docs/` with a status note (its numbers are illustrative, the app's are real); added a stance section (meaning, examples, role at each pipeline stage) and a stance tooltip in the app | Reviewed | DESIGN §1 *Stances*, DESIGN §8 *Visualization*, README |

## Summary

- **The developer owned the direction and the decisions:** the problem choice, dividing the
  work into subtasks, the core approach (semantic search + rubric LLM judge), the full
  pipeline design document, the visualization design, the rarity decay λ, the provider switch,
  and the review of every subtask before moving on.
- **The agent followed through on each subtask:** it investigated and measured, proposed
  suggestions, implemented the reviewed version, and verified it with tests.
- **Every design refinement is backed by measured evidence** and recorded in
  [DESIGN.md](DESIGN.md) §3 (changes 1–14). Evaluation used a frozen holdout set, and its
  results are reported unchanged.
