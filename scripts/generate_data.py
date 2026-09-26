"""Generate the synthetic corpus (50) and labeled golden set from data/angles.json.

Usage (from the repo root):
    ../myvenv/bin/python -m scripts.generate_data --dry-run     # print prompts, no API calls
    ../myvenv/bin/python -m scripts.generate_data               # generate both files
    ../myvenv/bin/python -m scripts.generate_data --only golden
    ../myvenv/bin/python -m scripts.generate_data --holdout     # frozen test set -> data/golden_holdout.json

Generated once and checked in, so tests never depend on a fresh generation (DESIGN.md §3).
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
GEN_MODEL = "gemini-3.5-flash-lite"  # corpus + dev set: one model for all items; see DESIGN.md §4
HOLDOUT_GEN_MODEL = "qwen/qwen3.8-27b"  # holdout: Gemini quota was exhausted; not the judge model
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
        lo, hi = plan["lengths"][rng.choice(list(plan["lengths"]))]
        slots.append({"persona": personas[i], "style": rng.choice(plan["styles"]),
                      "stance": stances[i % len(stances)], "words": f"{lo}-{hi}"})
    return slots


RULES = """Rules:
- Headline: 3-12 words. Body: the stated number of words (never more than 120).
- Vary vocabulary, sentence structure and examples; do not reuse phrasings across submissions.
- Stay faithful to each persona and style; the stance sets the tone toward the charge.
- Plain text, no hashtags or emojis. Never mention these instructions or personas."""


def angle_prompt(article: str, point: str, slots: list[dict]) -> str:
    lines = "\n".join(
        f"{i}. persona: {s['persona']}; style: {s['style']}; stance: {s['stance']}; body: {s['words']} words"
        for i, s in enumerate(slots, 1)
    )
    return f"""You are generating realistic reader comments on a local-news website for a research dataset.

ARTICLE:
{article}

Write {len(slots)} distinct reader submissions. Every submission must make this core point, and no other major argument:
"{point}"

Each one is written by a different reader:
{lines}

{RULES}

Return a JSON array of {len(slots)} objects with keys "headline" and "body", in the order listed."""


def offtopic_prompt(topic: str, slot: dict) -> str:
    return f"""You are generating a comment for a research dataset. It is posted in the comment section of a
local-news article about a downtown congestion charge, but it is about something else entirely.

Topic of the comment: {topic}

Write it as a {slot['persona']} in a {slot['style']} style; body {slot['words']} words. Make it specific and
original. Do not mention the article, cars, road traffic, or the city council.
{RULES}

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
            meta = [{"angle_id": a["id"]} | s for s in slots]
            corpus_jobs.append((angle_prompt(article, a["point"], slots), slots, meta))

    if which in ("all", "golden"):
        def single(category, angle_id, point, stances, prompt_fn=None):
            slots = make_slots(1, stances, plan, rng)
            lo, hi = gp[category]["expected"]
            meta = [{"category": category, "angle_id": angle_id, "expected_min": lo, "expected_max": hi} | slots[0]]
            prompt = prompt_fn(slots[0]) if prompt_fn else angle_prompt(article, point, slots)
            golden_jobs.append((prompt, slots, meta))

        # Categories are generated in plan order. "from" lists name a held-out or corpus angle.
        for cat, spec in gp.items():
            if "items" in spec:
                for item in spec["items"]:
                    base = angles[item["base"]]
                    point = f"{base['point']} As the main new contribution, add this specific idea: {item['new_detail']}"
                    single(cat, item["base"], point, base["stance"])
            elif "topics" in spec:
                for topic in spec["topics"]:
                    single(cat, None, topic, ["support", "oppose", "undecided"],
                           prompt_fn=lambda slot, topic=topic: offtopic_prompt(topic, slot))
            else:
                for aid in spec["from"]:
                    if aid in held:
                        single(cat, aid, held[aid]["point"], [held[aid]["stance"]])
                    else:
                        single(cat, aid, angles[aid]["point"], angles[aid]["stance"])

    return corpus_jobs, golden_jobs


def run_jobs(llm, jobs: list, prefix: str) -> list[dict]:
    rows = []
    for prompt, slots, meta in jobs:
        print(f"- {meta[0].get('category', meta[0].get('angle_id'))} ({len(slots)} item(s))")
        for sub, m in zip(generate(llm, prompt, slots), meta):
            rows.append({"id": f"{prefix}{len(rows) + 1:03d}", **sub.model_dump(mode="json", exclude={"id"}), **m,
                         "model": llm.last_model})
    return rows


def handwritten_rows(plan: dict, start: int) -> list[dict]:
    rows = []
    for i, h in enumerate(plan["golden_handwritten"]):
        sub = Submission(headline=h["headline"], body=h["body"], stance=h["stance"])
        lo, hi = h["expected"]
        rows.append({"id": f"g{start + i:03d}", **sub.model_dump(mode="json", exclude={"id"}),
                     "category": h["category"], "angle_id": None, "expected_min": lo, "expected_max": hi,
                     "persona": "handwritten"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["all", "corpus", "golden"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompts and counts, no API calls")
    ap.add_argument("--holdout", action="store_true", help="generate the frozen holdout test set instead")
    args = ap.parse_args()

    plan = json.loads((DATA / "angles.json").read_text())
    golden_file, seed = "golden.json", SEED
    if args.holdout:
        h = plan["holdout"]
        plan = {**plan, "held_out_angles": plan["held_out_angles"] + h["held_out_angles"],
                "golden_plan": h["golden_plan"], "golden_handwritten": h["golden_handwritten"]}
        golden_file, seed, args.only = "golden_holdout.json", h["seed"], "golden"
    article = load_fixed_content(DATA / "fixed_content.json").to_text()
    rng = random.Random(seed)
    corpus_jobs, golden_jobs = build_jobs(plan, article, args.only, rng)

    if args.dry_run:
        handwritten_rows(plan, 1)  # validates the handwritten items
        n_c = sum(len(j[1]) for j in corpus_jobs)
        n_g = sum(len(j[1]) for j in golden_jobs) + len(plan["golden_handwritten"])
        print(f"corpus: {n_c} items in {len(corpus_jobs)} calls | golden: {n_g} items "
              f"({len(golden_jobs)} calls + {len(plan['golden_handwritten'])} handwritten)\n")
        for jobs in (corpus_jobs, golden_jobs[-1:]):
            if jobs:
                print(jobs[0][0], "\n" + "=" * 80)
        return

    from novelty.llm import get_llm
    if args.holdout:  # written by a different model than the judge, so the judge never scores its own text
        llm = get_llm(model=os.getenv("GEN_MODEL", HOLDOUT_GEN_MODEL), fallbacks="", provider="groq")
    else:
        llm = get_llm(model=os.getenv("GEN_MODEL", GEN_MODEL), fallbacks="", provider="gemini")
    print(f"generation model: {llm.inner.cache_id}")

    if corpus_jobs:
        print("Generating corpus...")
        corpus = run_jobs(llm, corpus_jobs, "c")
        (DATA / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(corpus)} items to data/corpus.json")

    if golden_jobs:
        print("Generating golden set...")
        golden = run_jobs(llm, golden_jobs, "g")
        golden += handwritten_rows(plan, len(golden) + 1)
        prefix = "h" if args.holdout else "g"
        for row in golden:
            row["id"] = prefix + row["id"][1:]
        (DATA / golden_file).write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(golden)} items to data/{golden_file}")


if __name__ == "__main__":
    main()
