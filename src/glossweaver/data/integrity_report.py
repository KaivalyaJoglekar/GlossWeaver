"""Bounded near-duplicate audit; reports candidates without silently deleting data."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from glossweaver.data.split_integrity import lineage_keys, source_key
from glossweaver.utils import read_jsonl, sha256_file


def audit_records(records: list[dict[str, Any]], maximum_pairs: int = 100000) -> dict[str, Any]:
    owners = defaultdict(set)
    sources = Counter(source_key(str(r['gloss'])) for r in records)
    transforms = Counter()
    for row in records:
        for key in lineage_keys(row):
            owners[key].add(row.get('split', 'unknown'))
        value = row.get('operations', row.get('corruption_operations', []))
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = [value]
        transforms.update(value or [])
    # A bounded blocking heuristic: related sentences need overlapping first tokens.
    buckets = defaultdict(list)
    near = []
    checked = 0
    limited = False
    for row in records:
        text = source_key(str(row.get('source_sentence') or row.get('target') or row['gloss']))
        words = text.split()
        block = tuple(words[:2])
        for previous, other in buckets[block][-100:]:
            checked += 1
            if text != previous and SequenceMatcher(None, text, previous, autojunk=False).ratio() >= .92:
                near.append({'left_id': other['id'], 'right_id': row['id'], 'cross_split': other.get('split') != row.get('split')})
            if checked >= maximum_pairs:
                limited = True
                break
        if limited:
            break
        buckets[block].append((text, row))
    return {'audit_version': 'lineage-v2', 'examples': len(records), 'exact_duplicate_fragments': sum(n - 1 for n in sources.values()),
            'cross_split_lineage_keys': sorted(k for k, splits in owners.items() if len(splits) > 1),
            'near_duplicate_candidates': near[:500], 'near_duplicate_pairs_checked': checked,
            'near_duplicate_search_limited': limited, 'corruption_frequencies': dict(transforms),
            'coverage': dict(Counter(str(r.get('corpus_origin') or r.get('category') or 'unannotated') for r in records))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.files:
        if path.suffix == '.csv':
            import pandas as pd
            records = pd.read_csv(path).fillna('').to_dict('records')
        else:
            records = list(read_jsonl(path))
        rows.extend(dict(row, split=path.stem) for row in records)
    report = audit_records(rows)
    report['file_sha256'] = {str(p): sha256_file(p) for p in args.files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Audited {len(rows)} examples; {len(report['cross_split_lineage_keys'])} cross-split lineage keys")

if __name__ == '__main__':
    main()
