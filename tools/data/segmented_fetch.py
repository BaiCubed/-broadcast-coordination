#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import subprocess
import sys
from pathlib import Path


MINIMUM_PLAUSIBLE_SIZE = 1 << 20


def remote_size(url: str) -> int:
    proc = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True, check=True
    )
    lengths = [
        int(line.split(":", 1)[1].strip())
        for line in proc.stdout.splitlines()
        if line.lower().startswith("content-length")
    ]
    if not lengths:
        raise RuntimeError(f"no content-length for {url}")
    size = lengths[-1]
    if size < MINIMUM_PLAUSIBLE_SIZE:
        raise RuntimeError(
            f"implausible content-length {size} for {url}; check the URL points at the file body"
        )
    return size


def fetch_segment(url: str, start: int, stop: int, path: Path) -> tuple[Path, int]:
    want = stop - start + 1
    if path.exists():
        if path.stat().st_size == want:
            return path, want
        path.unlink()
    subprocess.run(
        ["curl", "-sS", "--fail", "--retry", "8", "--retry-delay", "5",
         "--connect-timeout", "30", "-r", f"{start}-{stop}", "-L", url, "--output", str(path)],
        check=True,
    )
    written = path.stat().st_size
    if written != want:
        raise RuntimeError(f"segment {path.name} got {written} bytes, expected {want}")
    return path, written


def digest(path: Path, algo: str) -> str:
    handle = hashlib.new(algo)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            handle.update(chunk)
    return handle.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finish a large download with parallel byte-range segments.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--algo", choices=["md5", "sha256", "none"], default="none")
    parser.add_argument("--checksum", default="")
    args = parser.parse_args()

    dest: Path = args.dest
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(args.url)
    have = part.stat().st_size if part.exists() else 0
    print(f"total={total} have={have} remaining={total - have}", flush=True)
    if have >= total:
        part.rename(dest)
        print("already complete", flush=True)
        return 0

    span = total - have
    count = max(1, min(args.segments, span // (1 << 20) or 1))
    size = span // count
    bounds = []
    for index in range(count):
        start = have + index * size
        stop = total - 1 if index == count - 1 else have + (index + 1) * size - 1
        bounds.append((start, stop, part.with_suffix(part.suffix + f".seg{index:02d}")))

    with cf.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(fetch_segment, args.url, *bound) for bound in bounds]
        for future in futures:
            path, written = future.result()
            print(f"segment {path.name} {written} bytes", flush=True)

    with part.open("ab") as sink:
        for _, _, segment in bounds:
            with segment.open("rb") as source:
                while True:
                    chunk = source.read(1 << 22)
                    if not chunk:
                        break
                    sink.write(chunk)
            segment.unlink()
    final = part.stat().st_size
    if final != total:
        print(f"SIZE MISMATCH assembled={final} expected={total}", flush=True)
        return 1
    if args.algo != "none" and args.checksum:
        actual = digest(part, args.algo)
        if actual != args.checksum:
            print(f"CHECKSUM MISMATCH {args.algo} expected={args.checksum} actual={actual}", flush=True)
            return 1
        print(f"checksum OK ({args.algo})", flush=True)
    part.rename(dest)
    print(f"OK {dest} {final} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
