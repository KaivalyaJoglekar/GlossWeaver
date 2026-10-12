#!/usr/bin/env python3
"""Create a jointly isolated study-v2 corpus without replacing historical datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from glossweaver.settings import ROOT
from glossweaver.utils import read_jsonl, write_jsonl, write_json, sha256_file
from glossweaver.data.split_integrity import group_split, assert_source_isolation, source_key
from glossweaver.data.grammar_labels import extract_grammar_labels
from glossweaver.data.integrity_report import audit_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'datasets/processed/study_v2')
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f'{args.output_root} already exists; choose a new --output-root to preserve dataset versions')
    accepted, quarantine, hashes = [], [], {}
    definitions = [('aslg_pc12', 'aslg_grouped', 'jsonl'), ('reconstruction', 'reconstruction_grouped', 'csv')]
    for source, destination, extension in definitions:
        for split in ('train', 'validation', 'test'):
            path = ROOT/'datasets/processed'/source/f'{split}.{extension}'
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
            records = read_jsonl(path) if extension == 'jsonl' else pd.read_csv(path).fillna('').to_dict('records')
            for row in records:
                if not any(c.isalpha() for c in str(row['gloss'])) or not str(row['target']).strip():
                    quarantine.append(dict(row, rejection_reason='empty_target_or_nonlexical_source'))
                    continue
                sentence_hash = hashlib.sha256(source_key(str(row['target'])).encode()).hexdigest()
                accepted.append(dict(row, id=f'{source}:{row["id"]}', source_record_id=row['id'],
                    study_corpus=destination, source_file=str(path.relative_to(ROOT)),
                    source_sentence_id=sentence_hash, source_sentence=row['target'], dataset_version='study-v2',
                    grammar_labels=extract_grammar_labels(str(row['gloss']), str(row['target']))))
    rebuilt = group_split(accepted, seed=42)
    groups = {s: [r for r in rebuilt if r['split'] == s] for s in ('train', 'validation', 'test')}
    assert_source_isolation(groups['train'], groups['validation'], groups['test'])
    report = audit_records(rebuilt)
    for _, destination, extension in definitions:
        output = args.output_root/destination
        output.mkdir(parents=True)
        counts = {}
        for split, rows in groups.items():
            selected = [row for row in rows if row['study_corpus'] == destination]
            counts[split] = len(selected)
            if extension == 'jsonl':
                write_jsonl(output/f'{split}.jsonl', selected)
            else:
                pd.DataFrame([dict(row, grammar_labels=json.dumps(row['grammar_labels'])) for row in selected]).to_csv(output/f'{split}.csv', index=False)
        write_json(output/'manifest.json', {'dataset_version': 'study-v2', 'split_version': 'joint-lineage-target-group-v2', 'seed': 42,
            'counts': counts, 'source_hashes': hashes, 'grammar_label_policy': os.environ.get('GLOSSWEAVER_GRAMMAR_PARSER', 'lexical-v1')})
        print(destination, counts)
    write_json(args.output_root/'audit.json', report)
    write_jsonl(args.output_root/'quarantine.jsonl', quarantine)

if __name__ == '__main__':
    main()
