"""
Download a GGUF model file from Hugging Face.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a GGUF model file.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to GGUF file.")
    parser.add_argument(
        "--output",
        default="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        help="Output path.",
    )
    return parser.parse_args()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urlopen(url) as response, open(output, "wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                print(f"\r{pct:5.1f}% ({downloaded/1e9:.2f} GB)", end="", flush=True)
    if total_bytes:
        print()
    print(f"Saved to {output}")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    try:
        download(args.url, output)
    except Exception as exc:
        print(f"Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
