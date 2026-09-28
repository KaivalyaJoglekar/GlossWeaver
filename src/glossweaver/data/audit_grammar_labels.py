from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from glossweaver.data.grammar_labels import LABEL_NAMES, explain_grammar_labels
from glossweaver.settings import ROOT


OPERATION_LABELS = {
    "ARTICLE_DROP": "ARTICLE",
    "PREPOSITION_DROP": "PREPOSITION",
    "AUXILIARY_DROP": "AUXILIARY",
    "OPTIONAL_PRONOUN_REDUCTION": "PRONOUN",
    "TENSE_NEUTRALIZATION": "TENSE_ASPECT",
    "AGREEMENT_SIMPLIFICATION": "AGREEMENT_INFLECTION",
    "WORD_ORDER": "WORD_ORDER",
}


def _aslg_rows(count: int, seed: int) -> pd.DataFrame:
    records = []
    path = ROOT / "datasets/processed/aslg_pc12/train.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return pd.DataFrame(records).sample(n=count, random_state=seed)


def audit_grammar_labels(count_per_source: int = 100, seed: int = 42) -> pd.DataFrame:
    aslg = _aslg_rows(count_per_source, seed).assign(audit_source="aslg_pc12", operations="")
    synthetic = pd.read_csv(ROOT / "datasets/processed/reconstruction/train.csv").sample(
        n=count_per_source, random_state=seed + 1
    ).assign(audit_source="synthetic_telegraphic")
    frame = pd.concat([aslg, synthetic], ignore_index=True)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        decisions = explain_grammar_labels(str(record["gloss"]), str(record["target"]))
        operations = set(str(record.get("operations", "")).split("|"))
        row: dict[str, object] = {
            "id": record["id"],
            "audit_source": record["audit_source"],
            "gloss": record["gloss"],
            "target": record["target"],
            "operations": record.get("operations", ""),
        }
        for decision in decisions:
            row[f"label_{decision.label.lower()}"] = decision.value
            row[f"reason_{decision.label.lower()}"] = " | ".join(decision.reasons)
            expected = int(any(OPERATION_LABELS.get(operation) == decision.label for operation in operations))
            row[f"operation_expected_{decision.label.lower()}"] = (
                expected if record["audit_source"] == "synthetic_telegraphic" else ""
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    output = ROOT / "results/data_audit"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "grammar_label_audit_200.csv", index=False)

    synthetic_audit = result[result["audit_source"] == "synthetic_telegraphic"]
    summary = []
    for label in LABEL_NAMES:
        actual = synthetic_audit[f"label_{label.lower()}"].astype(int)
        expected = synthetic_audit[f"operation_expected_{label.lower()}"].astype(int)
        summary.append({
            "label": label,
            "audited_pairs": len(synthetic_audit),
            "positive_rate": actual.mean(),
            "operation_expected_rate": expected.mean(),
            "agreement_with_known_operation": (actual == expected).mean(),
        })
    pd.DataFrame(summary).to_csv(output / "grammar_label_audit_summary.csv", index=False)
    return result


if __name__ == "__main__":
    audited = audit_grammar_labels()
    print(f"Audited {len(audited)} gloss/target pairs")
