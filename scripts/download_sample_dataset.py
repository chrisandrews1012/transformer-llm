#!/usr/bin/env python3
"""
Download a small sample dataset and convert it to JSONL format.

Uses only the Python standard library; no additional dataset libraries are required.

Example:
    python scripts/download_sample_dataset.py \
        --dataset tinyshakespeare \
        --output_path data/raw/tinyshakespeare/tinyshakespeare.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlopen


DATASET_SOURCES = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "description": "Tiny Shakespeare dialogue corpus",
    }
}


def download_text(url: str) -> str:
    """
    Download UTF-8 text from a URL.

    :param url: URL to fetch.
    :type url: str
    :returns: Decoded UTF-8 text content.
    :rtype: str
    """
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def group_lines(lines: List[str], group_size: int) -> Iterable[str]:
    """
    Group non-empty lines into fixed-size chunks joined by spaces.

    :param lines: Raw lines of text.
    :type lines: List[str]
    :param group_size: Number of non-empty lines per output chunk.
    :type group_size: int
    :returns: Iterator of joined text chunks.
    :rtype: Iterable[str]
    """
    buffer: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        buffer.append(line)
        if len(buffer) == group_size:
            yield " ".join(buffer)
            buffer = []

    if buffer:
        yield " ".join(buffer)


def write_jsonl(records: Iterable[str], output_path: Path, source_name: str) -> int:
    """
    Write text records to a JSONL file with a required ``text`` field.

    :param records: Text records to write.
    :type records: Iterable[str]
    :param output_path: Destination file path.
    :type output_path: Path
    :param source_name: Value for the ``source`` metadata field.
    :type source_name: str
    :returns: Number of records written.
    :rtype: int
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(records):
            record = {
                "text": text,
                "source": source_name,
                "example_id": idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    """
    Parse arguments, download the chosen dataset, and write it to JSONL.
    """
    parser = argparse.ArgumentParser(description="Download a small sample dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_SOURCES.keys()),
        default="tinyshakespeare",
        help="Which sample dataset to download",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/tinyshakespeare/tinyshakespeare.jsonl",
        help="Where to save the converted JSONL dataset",
    )
    parser.add_argument(
        "--group_lines",
        type=int,
        default=4,
        help="How many non-empty raw lines to merge into one JSONL example",
    )
    args = parser.parse_args()

    if args.group_lines <= 0:
        raise ValueError("--group_lines must be a positive integer")

    source = DATASET_SOURCES[args.dataset]
    output_path = Path(args.output_path)

    print("=" * 80)
    print(f"Downloading sample dataset: {args.dataset}")
    print(f"Description: {source['description']}")
    print(f"Source URL: {source['url']}")
    print(f"Output path: {output_path}")
    print(f"Grouping {args.group_lines} non-empty lines per example")
    print("=" * 80)

    raw_text = download_text(source["url"])
    raw_lines = raw_text.splitlines()
    records = list(group_lines(raw_lines, args.group_lines))
    count = write_jsonl(records, output_path, args.dataset)

    print(f"Downloaded {len(raw_lines)} raw lines")
    print(f"Wrote {count} JSONL examples to {output_path}")
    if records:
        print("\nFirst example:")
        print("-" * 80)
        print(records[0][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
