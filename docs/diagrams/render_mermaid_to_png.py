#!/usr/bin/env python3
"""
Render .mmd files to PNG using Kroki.io API.
POST diagram source to https://kroki.io/mermaid/png
"""
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

KROKI_URL = "https://kroki.io/mermaid/png"


def main():
    diagrams_dir = Path(__file__).resolve().parent
    mmd_files = sorted(diagrams_dir.glob("*.mmd"))
    if not mmd_files:
        print("No .mmd files in", diagrams_dir)
        return

    for mmd_path in mmd_files:
        code = mmd_path.read_text(encoding="utf-8").strip()
        if not code:
            continue
        png_path = mmd_path.with_suffix(".png")
        try:
            r = requests.post(
                KROKI_URL,
                data=code.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=60,
            )
            r.raise_for_status()
            png_path.write_bytes(r.content)
            print("OK", png_path.name)
        except Exception as e:
            print("FAIL", mmd_path.name, e)


if __name__ == "__main__":
    main()
