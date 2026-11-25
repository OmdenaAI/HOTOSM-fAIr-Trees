"""Collect every batch's ``relabel_boxes.json`` under a root into the one CSV the HOT converter reads.

Each tree keeps its offset box (the drawn box with the per-edge correction fitted on the gold pilot) in
EPSG:4326. usage: python -m experiments.relabel.export_boxes RELABEL_ROOT OUT_CSV
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

FIELDS = ("osm_id", "status", "west", "south", "east", "north")


def export(root: Path, out_csv: Path) -> Counter:
    """Write the CSV sorted by osm_id; return the status counts. An osm_id in two batches is an error."""
    rows = {}
    for path in sorted(Path(root).glob("**/relabel_boxes.json")):
        for rec in json.loads(path.read_text()):
            if rec["osm_id"] in rows:
                raise ValueError(f"{path}: osm_id {rec['osm_id']} is in an earlier batch too")
            bounds = rec["offset_lonlat"] if rec["status"] == "box" else ["", "", "", ""]
            rows[rec["osm_id"]] = [rec["osm_id"], rec["status"], *bounds]
    with Path(out_csv).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows[k] for k in sorted(rows))
    return Counter(r[1] for r in rows.values())


if __name__ == "__main__":
    counts = export(Path(sys.argv[1]), Path(sys.argv[2]))
    print(sum(counts.values()), "trees", dict(counts))
