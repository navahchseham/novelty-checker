"""Novelty Scorer: single entry point.

Usage (from the repo root):
    python main.py score --headline "..." --body "..." --stance oppose   # score one submission
    python main.py score --json submission.json [--add]                 # from a file; --add saves it to the pool
    python main.py evaluate [--set dev|holdout] [--consistency]         # success criteria report
    python main.py build-pool                                           # calibrate + rebuild the pool from the corpus
    python main.py generate [--holdout] [--dry-run]                     # regenerate the synthetic data
    python main.py lambda-sweep 2 1 0.4                                 # offline what-if for the rarity decay

Run `python main.py <command> --help` for a command's options.
Tests: `python -m pytest` (offline) and `python -m pytest -m live` (real models).
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "score": "scripts.score",
    "evaluate": "scripts.evaluate",
    "build-pool": "scripts.build_pool",
    "generate": "scripts.generate_data",
    "lambda-sweep": "scripts.lambda_sweep",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    command = sys.argv[1]
    sys.argv = [f"main.py {command}", *sys.argv[2:]]  # each script parses its own arguments
    importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()
