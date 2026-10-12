"""Blinded rating sheets and separate human summaries; never merge into auto metrics."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import pandas as pd
from glossweaver.utils import sha256_file, write_json

RATINGS = ('fluency', 'meaning_adequacy', 'grammatical_correctness', 'unsupported_content')


def prepare(files: list[Path], output: Path, seed: int = 42) -> None:
    if output.exists():
        raise ValueError('Output exists; choose a fresh directory')
    rows, mapping = [], []
    for file in files:
        for row in pd.read_csv(file).fillna('').to_dict('records'):
            rows.append({'source': row['gloss'], 'reference': row['reference'], 'prediction': row['prediction'],
                         'id': row['id'], 'model': row.get('model', ''), 'file': str(file)})
    random.Random(seed).shuffle(rows)
    sheets = []
    for index, row in enumerate(rows):
        identifier = f'item-{index+1:05d}'
        sheets.append({'item': identifier, 'source': row['source'], 'reference': row['reference'],
                       'prediction': row['prediction'], **{rating: '' for rating in RATINGS}, 'rater': '', 'notes': ''})
        mapping.append({'item': identifier, 'example_id': row['id'], 'model': row['model'], 'file': row['file']})
    output.mkdir(parents=True)
    pd.DataFrame(sheets).to_csv(output/'ratings.csv', index=False)
    write_json(output/'private_mapping.json', {'seed': seed, 'source_sha256': {str(f): sha256_file(f) for f in files}, 'items': mapping})
    (output/'INSTRUCTIONS.md').write_text('Rate fluency, meaning adequacy, and grammatical correctness from 1 (poor) to 5 (excellent). Rate unsupported content from 1 (none) to 5 (severe). Supply a rater ID. Keep private_mapping.json hidden from raters. Interpret incomplete source text conservatively; document ambiguity. Human ratings are not automatic metrics.\n')


def summarize(ratings: Path, output: Path) -> None:
    frame = pd.read_csv(ratings)
    for name in RATINGS:
        values = pd.to_numeric(frame[name], errors='raise')
        if values.isna().any() or not values.between(1, 5).all() or not (values % 1 == 0).all():
            raise ValueError(f'{name} requires completed integer ratings from 1 to 5')
        frame[name] = values
    if frame.rater.isna().any() or frame.rater.astype(str).str.strip().eq('').any():
        raise ValueError('All rows require a rater identifier')
    write_json(output, {'kind': 'human_evaluation', 'ratings_sha256': sha256_file(ratings),
                        'rows': len(frame), 'raters': int(frame.rater.nunique()),
                        'aggregate_means': {name: float(frame[name].mean()) for name in RATINGS},
                        'scope': 'Blinded sample aggregate; use private mapping for model analysis; no statistical superiority claim.'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('prepare'); make.add_argument('files', nargs='+', type=Path); make.add_argument('--output', required=True, type=Path); make.add_argument('--seed', type=int, default=42)
    collect = commands.add_parser('summarize'); collect.add_argument('ratings', type=Path); collect.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.files, args.output, args.seed)
    else:
        summarize(args.ratings, args.output)

if __name__ == '__main__':
    main()
