from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

from glossweaver.settings import ROOT
from glossweaver.utils import read_jsonl


CHALLENGE_TOKENS = (
    "MOTHER", "COOK", "FOOD", "TONIGHT", "WOMAN", "BUY", "CAR", "RED",
    "ME", "GO", "STORE", "YESTERDAY", "TEACHER", "STUDENT", "MATH",
    "CHILD", "PLAY", "OUTSIDE", "DOG", "RUN", "PARK",
)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _audit_row(record: dict[str, object]) -> dict[str, object]:
    gloss = str(record["gloss"])
    target = str(record["target"])
    gloss_tokens = _tokens(gloss)
    target_tokens = _tokens(target)
    overlap = len(set(gloss_tokens) & set(target_tokens)) / max(1, len(set(gloss_tokens)))
    return {
        "id": record["id"],
        "gloss": gloss,
        "target": target,
        "gloss_tokens": len(gloss_tokens),
        "target_tokens": len(target_tokens),
        "token_overlap": round(overlap, 6),
    }


def audit_dataset(
    dataset_file: str | Path = ROOT / "datasets/processed/aslg_pc12/train.jsonl",
    output_dir: str | Path = ROOT / "results/data_audit",
    *,
    seed: int = 42,
    audit_count: int = 500,
    saved_random_count: int = 100,
) -> dict[str, object]:
    records = list(read_jsonl(dataset_file))
    if len(records) < audit_count:
        raise ValueError(f"Need {audit_count} rows for audit; found {len(records)}")
    rng = random.Random(seed)
    sample = rng.sample(records, audit_count)
    rows = [_audit_row(record) for record in sample]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[:saved_random_count]).to_csv(output / "random_pairs.csv", index=False)

    token_set = set(CHALLENGE_TOKENS)
    matches: list[dict[str, object]] = []
    counts = {token: 0 for token in CHALLENGE_TOKENS}
    for record in records:
        present = sorted(set(token.upper() for token in _tokens(str(record["gloss"]))) & token_set)
        if present:
            for token in present:
                counts[token] += 1
            row = _audit_row(record)
            row["matched_tokens"] = "|".join(present)
            matches.append(row)
    pd.DataFrame(matches).to_csv(output / "challenge_token_matches.csv", index=False)

    frame = pd.DataFrame(rows)
    summary = {
        "dataset_rows": len(records),
        "audited_rows": len(rows),
        "mean_gloss_tokens": float(frame["gloss_tokens"].mean()),
        "mean_target_tokens": float(frame["target_tokens"].mean()),
        "mean_token_overlap": float(frame["token_overlap"].mean()),
        "challenge_token_counts": counts,
        "orientation": "gloss -> target English",
    }
    pd.DataFrame([{
        **{key: value for key, value in summary.items() if key != "challenge_token_counts"},
        **{f"count_{key.lower()}": value for key, value in counts.items()},
    }]).to_csv(output / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, max_rows=50))
    print("\nAudit summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ASLG-PC12 orientation and lexical coverage")
    parser.add_argument("--dataset-file", default=str(ROOT / "datasets/processed/aslg_pc12/train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/data_audit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit_dataset(args.dataset_file, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
