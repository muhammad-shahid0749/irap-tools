"""Cross-annotator evaluation of per-attribute precision, recall and F1.

Usage:
    python irap_vietnam_data_preparation/cross_annotator_eval.py <data_dir>

Reads (under ``<data_dir>/_raw/``, exactly like parse_coding_tables.py):
    coding-tables.zip                # OR an unzipped coding-tables/ directory
    attribute_metadata.json          # attribute_to_idx + value->irap mapping

Writes (into ``<data_dir>/_work/``):
    cross_annotator_report.json      # per-attribute P/R/F1, agreement, kappa,
                                     # and the concrete conflicting segments

What this measures:

Each coding-table file is treated as one annotator. ``parse_coding_tables.py``
normally keeps a single row per ``seg_id`` (the duplicate with the most filled
cells) and discards the rest; this tool instead uses the *pre-dedup* rows so a
segment coded by several annotators contributes one record per annotator, and
quantifies how much the annotators agree.

For every iRAP attribute independently, over the segments coded by >= 2
annotators (``--min-annotators``):

- ``MISSING_ATTR_CODE`` (-1, a blank cell) is treated as "no annotation" and
  excluded; a segment contributes to an attribute only where >= 2 annotators
  gave a genuine code for it.
- Precision/recall/F1 are macro-averaged over the iRAP codes *present* in the
  compared data (matching vidlu's ``only_present_classes=True`` convention),
  treating one annotator as truth and the other as prediction.
- ``pairs`` reports each *ordered* annotator pair A->B (truth=A, pred=B), where
  precision and recall are directional.
- ``pooled`` aggregates all ordered pairs into one confusion matrix. Pooling
  both directions makes that confusion symmetric, so pooled macro
  precision == recall == F1 -- a single symmetric agreement-style score;
  directional asymmetry shows up only in ``pairs``.
- ``pct_agreement`` and Cohen's ``cohen_kappa`` are symmetric and computed over
  the unordered annotation pairs (kappa is ``None`` when the codes show no
  variability, e.g. everyone agreed on one code).
- ``conflicts`` lists the actual segments whose genuine codes disagree, naming
  each annotator's source table and 1-based source row alongside its code.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import typing as T

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

import layout
from parse_coding_tables import (
    MISSING_ATTR_CODE,
    ParseSetupError,
    collect_all_rows,
    conflicting_attributes,
    format_row_ranges,
    num_filled_attribute_cells,
    prepare_parse_inputs,
)


def _is_genuine(code: T.Any) -> bool:
    """True if ``code`` is a real annotation (not blank/absent)."""
    return code not in (None, MISSING_ATTR_CODE)


def _clean_float(x: float | None) -> float | None:
    """Map NaN/None to None so the JSON report stays valid (no NaN tokens)."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# -----------------------------------------------------------------------------
# Pure metric helpers (operate on plain code arrays; no I/O)
# -----------------------------------------------------------------------------


def macro_prf(y_true: T.Sequence[int], y_pred: T.Sequence[int]) -> dict[str, float]:
    """Macro precision/recall/F1 over the codes present in ``y_true``/``y_pred``.

    Averaging only over present classes matches sklearn's ``average='macro'`` on
    the present-label subset and vidlu's ``only_present_classes=True``.
    """
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def agreement_kappa(a: T.Sequence[int], b: T.Sequence[int]) -> dict[str, T.Any]:
    """Raw percent agreement and Cohen's kappa for paired annotations ``a``/``b``."""
    n = len(a)
    if n == 0:
        return {"pct_agreement": None, "cohen_kappa": None, "n": 0}
    pct = float(np.mean(np.asarray(a) == np.asarray(b)))
    kappa = _clean_float(float(cohen_kappa_score(a, b)))
    return {"pct_agreement": pct, "cohen_kappa": kappa, "n": n}


# -----------------------------------------------------------------------------
# Data assembly
# -----------------------------------------------------------------------------


def group_by_segment(
    all_rows: T.Sequence[dict],
    attribute_names: T.Sequence[str],
) -> dict[str, dict[str, dict]]:
    """Group pre-dedup rows into ``{seg_id: {source_file: record}}``.

    A segment coded twice within one file collapses to the record with the most
    filled cells (same rule parse_coding_tables uses for cross-file dups), so
    each annotator contributes a single record per segment.
    """
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_rows:
        per_file = groups[rec["seg_id"]]
        source_file = rec["source_file"]
        prev = per_file.get(source_file)
        if prev is None or (
            num_filled_attribute_cells(rec, attribute_names)
            > num_filled_attribute_cells(prev, attribute_names)
        ):
            per_file[source_file] = rec
    return groups


def attribute_comparisons(
    shared: T.Mapping[str, dict[str, dict]],
    attr: str,
) -> T.Iterator[tuple[str, dict[str, int]]]:
    """Yield ``(seg_id, {source_file: code})`` of genuine codes for one attribute.

    Only segments where >= 2 annotators gave a genuine code for ``attr`` are
    yielded.
    """
    for seg_id, per_file in shared.items():
        coded = {
            source_file: rec.get(attr)
            for source_file, rec in per_file.items()
            if _is_genuine(rec.get(attr))
        }
        if len(coded) >= 2:
            yield seg_id, coded


def compute_attribute_metrics(
    shared: T.Mapping[str, dict[str, dict]],
    attr: str,
) -> dict | None:
    """Per-attribute pooled + per-ordered-pair metrics, or None if uncompared."""
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    unordered_a: list[int] = []
    unordered_b: list[int] = []
    pair_ordered: dict[tuple[str, str], tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    n_segments = 0
    for _seg_id, coded in attribute_comparisons(shared, attr):
        n_segments += 1
        files = sorted(coded)
        for a in files:
            for b in files:
                if a == b:
                    continue
                pooled_true.append(coded[a])
                pooled_pred.append(coded[b])
                t, p = pair_ordered[(a, b)]
                t.append(coded[a])
                p.append(coded[b])
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                unordered_a.append(coded[files[i]])
                unordered_b.append(coded[files[j]])

    if n_segments == 0:
        return None

    pooled = {**macro_prf(pooled_true, pooled_pred),
              **agreement_kappa(unordered_a, unordered_b)}
    pairs = {
        f"{a}->{b}": {**macro_prf(t, p), "n": len(t)}
        for (a, b), (t, p) in sorted(pair_ordered.items())
    }
    return {"n_compared_segments": n_segments, "pooled": pooled, "pairs": pairs}


def collect_conflicts(
    shared: T.Mapping[str, dict[str, dict]],
    attribute_names: T.Sequence[str],
    *,
    max_per_attribute: int,
) -> tuple[dict[str, list[dict]], Counter]:
    """Per-attribute listing of disagreeing segments with full provenance.

    Returns ``(conflicts_per_attr, counts)`` where each conflict entry is
    ``{seg_id, annotations: [{source_file, source_row, code}, ...]}`` (capped at
    ``max_per_attribute`` per attribute) and ``counts`` holds the full count.
    """
    conflicts_per_attr: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    for seg_id, per_file in shared.items():
        group = list(per_file.values())
        for attr in conflicting_attributes(group, attribute_names):
            counts[attr] += 1
            if len(conflicts_per_attr[attr]) >= max_per_attribute:
                continue
            annotations = [
                {
                    "source_file": rec["source_file"],
                    "source_row": rec["source_row"],
                    "code": rec.get(attr),
                }
                for rec in sorted(
                    group, key=lambda r: (r["source_file"], r["source_row"])
                )
                if _is_genuine(rec.get(attr))
            ]
            conflicts_per_attr[attr].append(
                {"seg_id": seg_id, "annotations": annotations}
            )
    return conflicts_per_attr, counts


def shared_segment_ranges(
    shared: T.Mapping[str, dict[str, dict]],
) -> list[dict]:
    """Summarise the duplicated segments as seg_id ranges per annotator set.

    Groups the shared segments by the exact set of annotators (source files)
    that coded them and collapses each group's seg_ids into continuous ranges
    (the same ``format_row_ranges`` rendering parse_coding_tables uses for its
    duplicate warnings). Returns one entry per annotator set, most segments
    first.
    """
    by_set: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for seg_id, per_file in shared.items():
        by_set[tuple(sorted(per_file))].append(int(seg_id))
    entries = [
        {
            "annotators": list(annotators),
            "count": len(seg_ids),
            "seg_id_ranges": format_row_ranges(seg_ids),
        }
        for annotators, seg_ids in by_set.items()
    ]
    entries.sort(key=lambda e: (-e["count"], e["annotators"]))
    return entries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="IRAP_Vietnam dataset root.")
    parser.add_argument(
        "--min-annotators", type=int, default=2,
        help="Only evaluate segments coded by at least this many annotators "
             "(distinct source files). Default: 2.",
    )
    parser.add_argument(
        "--max-conflicts-per-attribute", type=int, default=200,
        help="Cap the conflicting-segment listing per attribute (the full count "
             "is still reported). Default: 200.",
    )
    args = parser.parse_args(argv)

    if args.min_annotators < 2:
        print("ERROR: --min-annotators must be >= 2.", file=sys.stderr)
        return 2

    try:
        inputs = prepare_parse_inputs(args.data_dir)
    except ParseSetupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.exit_code

    all_rows = collect_all_rows(inputs)
    attribute_names = inputs.attribute_names

    groups = group_by_segment(all_rows, attribute_names)
    shared = {
        seg_id: per_file
        for seg_id, per_file in groups.items()
        if len(per_file) >= args.min_annotators
    }
    annotators = sorted({rec["source_file"] for rec in all_rows})

    conflicts_per_attr, conflict_counts = collect_conflicts(
        shared, attribute_names, max_per_attribute=args.max_conflicts_per_attribute
    )

    attributes: dict[str, dict] = {}
    for attr in attribute_names:
        entry = compute_attribute_metrics(shared, attr)
        if entry is None:
            continue
        entry["num_conflicts"] = int(conflict_counts.get(attr, 0))
        if conflicts_per_attr.get(attr):
            entry["conflicts"] = conflicts_per_attr[attr]
        attributes[attr] = entry

    out = layout.work_dir(args.data_dir)
    report = {
        "meta": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "coding_tables_dir": str(
                inputs.coding_tables_dir.relative_to(args.data_dir)
            ),
            "attribute_metadata_path": str(
                layout.attr_meta_path(args.data_dir).relative_to(args.data_dir)
            ),
            "min_annotators": args.min_annotators,
        },
        "summary": {
            "annotators": annotators,
            "num_segments_total": len(groups),
            "num_shared_segments": len(shared),
            "num_attributes_compared": len(attributes),
            "shared_segments_by_annotator_set": shared_segment_ranges(shared),
        },
        "attributes": attributes,
    }

    out.mkdir(parents=True, exist_ok=True)
    report_path = layout.cross_annotator_report_path(args.data_dir)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False))

    _print_summary(report)
    print(f"Wrote {report_path}")
    return 0


def _print_summary(report: dict) -> None:
    """Compact stdout summary, attributes ordered by worst agreement first."""
    summary = report["summary"]
    print(f"Annotators (files contributing rows): {len(summary['annotators'])}")
    print(f"Segments total: {summary['num_segments_total']}; shared by >= "
          f"{report['meta']['min_annotators']} annotator(s): "
          f"{summary['num_shared_segments']}")
    shared_sets = summary["shared_segments_by_annotator_set"]
    if shared_sets:
        print("Shared (duplicate) segments by annotator set:")
        for entry in shared_sets:
            print(f"  {' + '.join(entry['annotators'])}: "
                  f"{entry['count']} segment(s) — seg_ids {entry['seg_id_ranges']}")
    attributes: dict[str, dict] = report["attributes"]
    if not attributes:
        print("No attribute had segments coded by multiple annotators.")
        return

    def sort_key(item: tuple[str, dict]) -> tuple[float, str]:
        kappa = item[1]["pooled"]["cohen_kappa"]
        return (kappa if kappa is not None else math.inf, item[0])

    headers = ("Attribute", "F1", "kappa", "agree", "segments", "conflicts")
    aligns = ("<", ">", ">", ">", ">", ">")
    rows: list[tuple[str, ...]] = []
    for attr, entry in sorted(attributes.items(), key=sort_key):
        pooled = entry["pooled"]
        kappa = pooled["cohen_kappa"]
        rows.append((
            attr,
            f"{pooled['macro_f1']:.3f}",
            "n/a" if kappa is None else f"{kappa:.3f}",
            f"{pooled['pct_agreement']:.1%}",
            str(entry["n_compared_segments"]),
            str(entry["num_conflicts"]),
        ))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells: T.Sequence[str]) -> str:
        return "  " + "  ".join(
            f"{c:{a}{w}}" for c, a, w in zip(cells, aligns, widths)
        )

    print("Per-attribute pooled agreement (worst Cohen's kappa first):")
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    sys.exit(main())
