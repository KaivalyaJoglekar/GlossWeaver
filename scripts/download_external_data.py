#!/usr/bin/env python3
"""Acquire only text/metadata columns from the official 2M-Flores-ASL release."""
from __future__ import annotations
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glossweaver.settings import ROOT
from glossweaver.utils import write_json, write_jsonl, sha256_file
from glossweaver.data.flores_asl import prepare_flores

DATASET = "facebook/2M-Flores-ASL"


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "GlossWeaver-research/0.1"}), timeout=90) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Preprocess an existing downloaded text export")
    args = parser.parse_args()
    raw = ROOT / "datasets/raw/2m_flores_asl"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        release = fetch(f"https://huggingface.co/api/datasets/{DATASET}")
        revision = release["sha"]
        params = {"dataset": DATASET}
        info = fetch("https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode(params))
        acquired = {}
        for spec in info["splits"]:
            split = spec["split"]
            if split not in {"dev", "devtest"}:
                raise ValueError(f"Unexpected official split {split}")
            records, offset = [], 0
            while True:
                query = dict(params, config=spec["config"], split=split, offset=offset, length=100)
                page = fetch("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query))
                for item in page["rows"]:
                    # Video content and URLs to video storage are intentionally omitted.
                    row = item["row"]
                    records.append({key: row.get(key) for key in ("id", "URL", "domain", "topic", "sentence", "gloss", "signer")})
                offset += len(page["rows"])
                if offset >= page["num_rows_total"]:
                    break
                if not page["rows"]:
                    raise ValueError("Incomplete dataset-server response")
            path = raw / f"{split}.jsonl"
            write_jsonl(path, records)
            acquired[split] = {"rows": len(records), "sha256": sha256_file(path)}
            print(f"Acquired {split}: {len(records)} text rows", flush=True)
        # The viewer serves current release data: check that it did not change mid-download.
        if fetch(f"https://huggingface.co/api/datasets/{DATASET}")["sha"] != revision:
            raise ValueError("Source revision changed during acquisition; retry")
        write_json(raw / "provenance.json", {
            "dataset": DATASET, "revision": revision, "license": "CC-BY-SA-4.0",
            "source": f"https://huggingface.co/datasets/{DATASET}",
            "acquisition": "official dataset-server text projection; video omitted",
            "annotation": "human glosses, expert harmonization per official dataset card",
            "splits": acquired,
        })
    print(prepare_flores(raw, ROOT / "datasets/processed/2m_flores_asl"))

if __name__ == "__main__":
    main()
