"""Download all files from a Seafile public share link.

Supports resume via HTTP Range and skips files whose local size already matches
the remote size. Recurses into subdirectories.

Usage:
    python download_images.py <data_dir> <share_url>

Writes to ``<data_dir>/_raw/image_rars/``.
"""
import argparse
import json
import re
import sys
import typing as T
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

import layout

CHUNK_SIZE = 1 << 20  # 1 MiB


def parse_share_url(share_url: str) -> tuple[str, str]:
    """Return (server_root, token) from a share link like https://host/d/<token>/."""
    m = re.match(r"^(https?://[^/]+)/d/([0-9a-f]+)/?", share_url)
    if not m:
        raise ValueError(f"Not a Seafile share link: {share_url!r}")
    return m.group(1), m.group(2)


def list_dirents(server: str, token: str, path: str = "/") -> list[dict]:
    """List entries directly under `path` on the share."""
    qs = {"path": path}
    url = f"{server}/api/v2.1/share-links/{token}/dirents/?{urllib.parse.urlencode(qs)}"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    return data["dirent_list"]


def walk_files(server: str, token: str) -> T.Iterator[dict]:
    """Yield file dirents from the share, recursing into folders."""
    stack = ["/"]
    while stack:
        path = stack.pop()
        for entry in list_dirents(server, token, path):
            if entry["is_dir"]:
                stack.append(entry["folder_path"])
            else:
                yield entry


def download_file(server: str, token: str, remote_path: str, dest: Path,
                  expected_size: int) -> None:
    """Download `remote_path` to `dest`, resuming if a partial file exists."""
    qs = {"p": remote_path, "dl": "1"}
    url = f"{server}/d/{token}/files/?{urllib.parse.urlencode(qs)}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing == expected_size:
        print(f"[skip] {remote_path} ({expected_size} bytes already present)")
        return
    if existing > expected_size:
        # Local file is larger than remote – likely corrupt; restart.
        print(f"[warn] {dest} larger than remote ({existing} > {expected_size}); "
              f"restarting download")
        dest.unlink()
        existing = 0

    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[resume] {remote_path} from byte {existing}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, mode) as f:
        with tqdm(
                total=expected_size, initial=existing, unit="B", unit_scale=True,
                unit_divisor=1024, desc=remote_path.lstrip("/"), miniters=1,
        ) as bar:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    final = dest.stat().st_size
    if final != expected_size:
        raise RuntimeError(
            f"Size mismatch for {remote_path}: got {final}, expected {expected_size}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="iRAP-Vietnam dataset root.")
    parser.add_argument("share_url",
                        help="Seafile public share link, https://host/d/<token>/.")
    args = parser.parse_args(argv)

    server, token = parse_share_url(args.share_url)
    out = layout.rars_dir(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = list(walk_files(server, token))
    total = sum(f["size"] for f in files)
    print(f"Found {len(files)} file(s), total {total / 1e9:.2f} GB")
    for f in files:
        print(f"  {f['size']:>14}  {f['file_path']}")

    for f in files:
        rel = f["file_path"].lstrip("/")
        dest = out / rel
        download_file(server, token, f["file_path"], dest, f["size"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
