"""Manual check of the WebGIS sidecar files produced by cutting a video.

Prints the first and last timestamped points of the original sidecar and of each cut, and the
LINESTRING of the first cut's SQL file, for comparison by eye. Run from the repository root
after cutting with `webgis_cuts.txt`:

    python -m irap_video_cutting.webgis --input-dir input_webgis --output-dir output_webgis_test \
        --list-file packages/irap_video_cutting/examples/webgis_cuts.txt
    python packages/irap_video_cutting/examples/check_webgis_cuts.py
"""

import json
from pathlib import Path

STEM = "VID_20250104_085135_00_008_20251122224908"
INPUT_DIR = Path("input_webgis")
OUTPUT_DIR = Path("output_webgis_test")
CUT_TITLES = ["0-20s", "20-40s (re-zeroed)", "40-60s (re-zeroed)", "60-95.7s (re-zeroed)"]


def print_endpoints(title: str, path: Path) -> None:
    points = [p for p in json.loads(path.read_text()) if "time" in p]
    print(f"\n=== {title} ===")
    print(f"First point: time={points[0]['time']}, coords={points[0]['coordinates']}")
    print(f"Last point: time={points[-1]['time']}, coords={points[-1]['coordinates']}")


print_endpoints("Original video", INPUT_DIR / f"{STEM}.json")
for i, title in enumerate(CUT_TITLES, 1):
    print_endpoints(f"Cut {title}", OUTPUT_DIR / f"{STEM}_{i}.json")

print("\n=== SQL boundary coordinates ===")
sql = (OUTPUT_DIR / f"{STEM}_1.sql").read_text()
start = sql.find("LINESTRING(") + len("LINESTRING(")
coords = sql[start:sql.find(")", start)].split(",")
print(f"Segment 1 SQL has {len(coords)} coordinate pairs")
print(f"First: {coords[0]}")
print(f"Last: {coords[-1]}")
