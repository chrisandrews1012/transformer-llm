#!/usr/bin/env python3
"""
Optional SFT stub.

This script exists so users get a clear message instead of a missing-file error.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """
    Print an informational message and exit; SFT training is not yet implemented.
    """
    parser = argparse.ArgumentParser(
        description="Optional SFT entry point. This repo does not ship a full SFT training scaffold."
    )
    parser.add_argument("--config", type=str, help="Reference-only SFT config path")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        help="Base checkpoint path for experimentation",
    )
    parser.parse_args()

    print(
        "This repo does not currently include a full SFT trainer.\n"
        "You may use scripts/download_sft_dataset.py to inspect a small SFT-style dataset."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
