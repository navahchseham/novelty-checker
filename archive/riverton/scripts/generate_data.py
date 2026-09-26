"""Generate the synthetic corpus (~50) and labeled golden set from data/angles.json.

Usage (from the repo root):
    ../myvenv/bin/python -m scripts.generate_data --dry-run     # print prompts, no API calls
    ../myvenv/bin/python -m scripts.generate_data               # generate both files
    ../myvenv/bin/python -m scripts.generate_data --only golden

See DESIGN.md §3 for the rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from pydantic import BaseModel, ValidationError

from novelty.schema import Submission, load_fixed_content

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
GEN_MODEL = "gemini-3.5-flash-lite"  # 3.8-flash free-tier quota was exhausted; see PLAN.md log
TEMPERATURE = 1.0
MAX_ATTEMPTS = 3


class Draft(BaseModel):
    headline: str
    body: str


def make_slots(n: int, stances: list[str], plan: dict, rng: random.Random) -> list[dict]:
    """Give each of n submissions a distinct persona and a style, length and stance."""
    personas = rng.sample(plan["personas"], n)
    slots = []
    for i in range(n):
        length = rng.choice(list(plan["lengths"]))
        lo, hi = plan["lengths"][length]
        slots.append({
            "persona": personas[i],
            "style": rng.choice(plan["styles"]),
            "stance": stances[i % len(stances)],
            "words": f"{lo}-{hi}",
        })
    return slots


def angle_prompt(article: str, point: str, slots: list[dict]) -> str:
    lines = "\n".join(
        f"{i}. persona: {s['persona']}; style: {s['style']}; stance on the pilot: {s['stance']}; "
        f"body length: {s['words']} words"
        for i, s in enumerate(slots, 1)
    )
    return f"""You are generating realistic reader comments on a local-news website for a research dataset.

ARTICLE:
{article}

Write {len(slots)} distinct reader submissions. Every submission must make this core point, and no other major argument:
"{point}"

Each one is written by a different reader:
{lines}

Rules:
- Vary vocabulary, sentence structure and examples; do not reuse phrasings across submissions.
- Stay faithful to each persona and style; the stance sets the tone toward the pilot.
- Headline: at most 12 words. Plain text, no hashtags or emojis.
- Never mention these instructions, personas, or that the text is generated.

Return a JSON array of {len(slots)} objects with keys "headline" and "body", in the order listed."""


def offtopic_prompt(topic: str, slot: dict) -> str:
    return f"""You are generating a reader comment for a research dataset. It is posted under a local-news
article about a city four-day work week, but it ignores the article completely.

Topic of the comment: {topic}

Write it as a {slot['persona']} in a {slot['style']} style, {slot['words']} words. Make it thoughtful,
specific and original. Do not mention the article, work weeks, schedules or city employees.
Headline: at most 12 words. Plain text, no hashtags or emojis.

Return a JSON array with one object with keys "headline" and "body"."""


def generate(llm, prompt: str, slots: list[dict]) -> list[Submission]:
    """Call the LLM and validate every draft against the Submission schema, retrying on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        drafts = llm.generate_json(prompt, list[Draft], temperature=TEMPERATURE)
        try:
            if len(drafts) != len(slots):
                raise ValueError(f"expected {len(slots)} drafts, got {len(drafts)}")
            return [Submission(headline=d["headline"], body=d["body"], stance=s["stance"])
                    for d, s in zip(drafts, slots)]
        except (ValueError, ValidationError) as e:
            print(f"  attempt {attempt} rejected: {str(e).splitlines()[0]}")
            if hasattr(llm, "invalidate"):
                llm.invalidate(prompt, list[Draft], TEMPERATURE)
    raise RuntimeError("generation failed validation after retries")


def build_jobs(plan: dict, article: str, which: str, rng: random.Random) -> tuple[list, list]:
    """Return (corpus_jobs, golden_jobs). Each job: (prompt, slots, metadata-per-slot)."""
    corpus_jobs, golden_jobs = [], []
    angles = {a["id"]: a for a in plan["corpus_angles"]}
    held = {a["id"]: a for a in plan["held_out_angles"]}
    gp = plan["golden_plan"]

    if which in ("all", "corpus"):
        for a in plan["corpus_angles"]:
            slots = make_slots(a["count"], a["stance"], plan, rng)
            meta = [{"angle_id": a["id"], "tier": a["tier"]} | s for s in slots]
            corpus_jobs.append((angle_prompt(article, a["point"], slots), slots, meta))

    if which in ("all", "golden"):
        def single(category, angle_id, point, stances):
            slots = make_slots(1, stances, plan, rng)
            lo, hi = gp[category]["expected"]
            meta = [{"category": category, "angle_id": angle_id, "expected_min": lo, "expected_max": hi} | slots[0]]
            golden_jobs.append((angle_prompt(article, point, slots), slots, meta))

        for hid in gp["novel_relevant"]["from"]:
            single("novel_relevant", hid, held[hid]["point"], [held[hid]["stance"]])
        for item in gp["partly_novel"]["items"]:
            base = angles[item["base"]]
            point = f"{base['point']} As the main new contribution, add this specific idea: {item['new_detail']}"
            single("partly_novel", item["base"], point, base["stance"])
        for cat in ("paraphrase_common", "paraphrase_rare"):
            for aid in gp[cat]["from"]:
                single(cat, aid, angles[aid]["point"], angles[aid]["stance"])
        lo, hi = gp["offtopic_novel"]["expected"]
        for topic in gp["offtopic_novel"]["topics"]:
            slots = make_slots(1, [rng.choice(["support", "oppose", "mixed"])], plan, rng)
            meta = [{"category": "offtopic_novel", "angle_id": None, "expected_min": lo, "expected_max": hi,
                     "topic": topic} | slots[0]]
            golden_jobs.append((offtopic_prompt(topic, slots[0]), slots, meta))

    return corpus_jobs, golden_jobs


def run_jobs(llm, jobs: list, prefix: str) -> list[dict]:
    rows = []
    for prompt, slots, meta in jobs:
        print(f"- {meta[0].get('category', meta[0].get('angle_id'))} ({len(slots)} item(s))")
        subs = generate(llm, prompt, slots)
        for sub, m in zip(subs, meta):
            rows.append({"id": f"{prefix}{len(rows) + 1:03d}", **sub.model_dump(mode="json", exclude={"id"}), **m,
                         "model": llm.last_model})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["all", "corpus", "golden"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompts and counts, no API calls")
    args = ap.parse_args()

    plan = json.loads((DATA / "angles.json").read_text())
    article = load_fixed_content(DATA / "fixed_content.json").to_text()
    rng = random.Random(SEED)
    corpus_jobs, golden_jobs = build_jobs(plan, article, args.only, rng)

    if args.dry_run:
        n_c = sum(len(j[1]) for j in corpus_jobs)
        n_g = sum(len(j[1]) for j in golden_jobs) + len(plan["golden_handwritten"])
        print(f"corpus: {n_c} items in {len(corpus_jobs)} calls | golden: {n_g} items "
              f"({len(golden_jobs)} calls + {len(plan['golden_handwritten'])} handwritten)\n")
        for jobs in (corpus_jobs, golden_jobs):
            if jobs:
                print(jobs[0][0], "\n" + "=" * 80)
        return

    from novelty.llm import get_llm
    # One model for the whole dataset, so embeddings don't pick up model-specific writing style.
    llm = get_llm(model=os.getenv("GEN_MODEL", GEN_MODEL), fallbacks="")
    print(f"generation model: {llm.inner.cache_id}")

    if corpus_jobs:
        print("Generating corpus...")
        corpus = run_jobs(llm, corpus_jobs, "c")
        (DATA / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(corpus)} items to data/corpus.json")

    if golden_jobs:
        print("Generating golden set...")
        golden = run_jobs(llm, golden_jobs, "g")
        gp = plan["golden_plan"]
        for h in plan["golden_handwritten"]:
            sub = Submission(headline=h["headline"], body=h["body"], stance=h["stance"])
            lo, hi = gp[h["category"]]["expected"]
            golden.append({"id": f"g{len(golden) + 1:03d}", **sub.model_dump(mode="json", exclude={"id"}),
                           "category": h["category"], "angle_id": None, "expected_min": lo, "expected_max": hi,
                           "persona": "handwritten"})
        (DATA / "golden.json").write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(golden)} items to data/golden.json")


if __name__ == "__main__":
    main()
