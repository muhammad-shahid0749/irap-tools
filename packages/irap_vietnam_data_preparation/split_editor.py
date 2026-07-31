"""Interactive map GUI for manually assigning sequences to train/val/test splits.

Requires Streamlit >= 1.35 (for on_select="rerun").

Launch:
    streamlit run irap_vietnam_data_preparation/split_editor.py -- <data_dir>

Reads ``<data_dir>/{segment_id_to_road_data,road_id_to_segment_id_sequence}.json``;
writes ``<data_dir>/splits.json``.

Interaction:
    1. Pick an active split in the sidebar (Train / Val / Test / None).
    2. Draw a rectangle on the map – all sequences whose centroid falls
       inside are assigned to the active split.
    3. Undo reverts the last assignment batch; Reset clears all.
    4. Save writes splits.json (format documented in README.md, "Output format").

The sidebar reports class coverage per split: how many (attribute, IRAP code)
classes are absent, and how many fall short of a minimum number of *sequences*
(not segments – consecutive segments of one road are near-duplicates). Classes
too rare to appear in every split under any assignment are separated out, so
the headline figure tracks the assignment rather than the dataset.

An existing splits.json is loaded as the starting assignment, so the editor can
be re-opened after build_metadata.py was re-run on new parsed data. Because the
file stores seg_ids while the editor assigns whole sequences, any way in which
the two have drifted apart (dropped seg_ids, sequences merged or split by the
adjacency check, a stale unlabeled_unlocated_segment_ids.json) is resolved
deterministically and reported in the sidebar rather than applied silently.
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import layout


# Colours (one per split label)

SPLITS = ("train", "val", "test", "none")
REAL_SPLITS = tuple(s for s in SPLITS if s != "none")
COLORS: dict[str, str] = {
    "train": "#4477AA",
    "val": "#CC3333",
    "test": "#559933",
    "none": "#BBBBBB",
}

# Class-coverage settings. A "class" is one (attribute, IRAP code) pair.

# Written by parse_coding_tables.py into cells that had no annotation; it is an
# ignore label, not a class, so it never contributes to any count here.
MISSING_ATTR_CODE = -1

# Support is counted in distinct *sequences*, not segments: consecutive 20 m
# segments of one road carry near-identical attributes, so 500 segments of a
# rare class drawn from a single section are one example, not 500.
DEFAULT_MIN_SEQ = 3

# A class needs ``n_min`` sequences in each of the three splits, so unless it
# occurs in at least this many sequences dataset-wide no assignment of whole
# sequences can cover it everywhere. Such classes are reported separately rather
# than counted against the current assignment, which would otherwise show a
# floor that no amount of redrawing can bring down.
STRATIFIABLE_FACTOR = len(REAL_SPLITS)

COVERAGE_OK_COLOR = "#228833"
COVERAGE_WARN_COLOR = "#CC5500"


def _muted(hex_color: str, t: float = 0.5) -> str:
    """Interpolate ``hex_color`` toward grey ``#BBBBBB`` by factor ``t``."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    gr = 0xBB
    r = round(r * (1 - t) + gr * t)
    g = round(g * (1 - t) + gr * t)
    b = round(b * (1 - t) + gr * t)
    return f"#{r:02X}{g:02X}{b:02X}"


UNLABELED_COLORS: dict[str, str] = {s: _muted(c) for s, c in COLORS.items()}

# Metadata files rewritten by build_metadata.py. Their (mtime, size) forms the
# cache key of the loaders below, so re-running the build invalidates them.
METADATA_FILENAMES: tuple[str, ...] = (
    "segment_id_to_road_data.json",
    "road_id_to_segment_id_sequence.json",
    layout.UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME,
    layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME,
    layout.ATTR_META_FILENAME,
)


@dataclass
class Layer:
    """A category of sequences (labeled or unlabeled) rendered as one centroid trace."""

    name: str  # "labeled" | "unlabeled"
    data: dict[str, dict]
    colors: dict[str, str]
    marker_size: int
    key_prefix: str  # "" or "unlabeled_"
    state_key: str  # st.session_state key holding sequence -> split


# CLI args (streamlit passes everything after "--" as sys.argv)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="IRAP_Vietnam dataset root.")
    args = parser.parse_args()
    args.metadata_dir = layout.metadata_dir(args.data_dir)
    args.output = args.metadata_dir / "splits.json"
    return args


# Data loading (cached so the large JSON is only read once per session)


def _metadata_fingerprint(metadata_dir: Path) -> tuple:
    """Return an (mtime, size) fingerprint of the build_metadata.py outputs.

    ``st.cache_data`` keys on the argument values and caches process-wide, so
    without this the loaders below would keep serving the metadata read at
    server start even after ``build_metadata.py`` was re-run and the browser
    refreshed – and a subsequent Save would write splits.json from obsolete
    sequence membership.
    """
    fingerprint = []
    for name in METADATA_FILENAMES:
        path = metadata_dir / name
        try:
            stat = path.stat()
        except OSError:
            fingerprint.append((name, None, None))
        else:
            fingerprint.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


@st.cache_data
def load_unlabeled_sequence_data(metadata_dir: str, fingerprint: tuple) -> dict[str, dict]:
    """Return ``{sequence_id: {"segs": [seg_id, ...], "centroid": (lat, lon)}}``.

    ``fingerprint`` is unused; it is part of the cache key so that a rebuild of
    the metadata invalidates this cache. Returns ``{}`` if the file does not
    exist.
    """
    path = Path(metadata_dir) / layout.UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        seq: {
            "segs": list(entry["segs"]),
            "centroid": (float(entry["centroid"][0]), float(entry["centroid"][1])),
        }
        for seq, entry in raw.items()
    }


@st.cache_data
def load_unlocated_seg_ids(metadata_dir: str, fingerprint: tuple) -> list[str]:
    """Return unlabeled seg_ids with no derivable map location.

    These come from image folders that have no labeled siblings; they are
    auto-assigned to the ``unlabeled_unlocated`` split. ``fingerprint`` is
    unused; see :func:`load_unlabeled_sequence_data`. Returns ``[]`` if the file
    does not exist.
    """
    path = Path(metadata_dir) / layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


@st.cache_data
def load_sequence_data(
    metadata_dir: str,
    fingerprint: tuple,
) -> tuple[dict, dict]:
    """Return (sequence_data, seg_to_sequence).

    sequence_data[sequence] = {
        "segs": [seg_id, ...],          # ordered
        "coords": [(lat, lon), ...],    # parallel to segs
        "centroid": (lat, lon),
        "classes": {(attribute, code): n_segments, ...},
    }
    seg_to_sequence[seg_id] = sequence

    Sequences are the road_ids of ``road_id_to_segment_id_sequence.json``, *not*
    the ``section`` field of ``segment_id_to_road_data.json``: a section whose
    seg_id/distance adjacency is inconsistent is emitted as several
    ``"<section>__partN"`` road_ids. ``seg_to_sequence`` is therefore built by
    inverting ``sequence_data``, so it is the exact inverse of what the save
    step writes.

    ``classes`` is the sequence's histogram over the ``required_attributes``
    already present in ``segment_id_to_road_data.json``, and is what the class
    coverage report is summed from. :data:`MISSING_ATTR_CODE` and ``None`` are
    excluded, so an attribute that was never coded contributes no classes at all
    rather than a spurious "missing" one.
    """
    base = Path(metadata_dir)
    with open(base / "segment_id_to_road_data.json", encoding="utf-8") as f:
        seg_to_road: dict = json.load(f)
    with open(base / "road_id_to_segment_id_sequence.json", encoding="utf-8") as f:
        road_to_segs: dict = json.load(f)

    sequence_data: dict = {}
    for seq, segs in road_to_segs.items():
        coords = [(seg_to_road[sid]["lat"], seg_to_road[sid]["lon"]) for sid in segs if sid in seg_to_road]
        if not coords:
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        kept = [s for s in segs if s in seg_to_road]
        classes: Counter = Counter()
        for sid in kept:
            for attr, code in seg_to_road[sid].get("required_attributes", {}).items():
                if code is None or int(code) == MISSING_ATTR_CODE:
                    continue
                classes[(attr, int(code))] += 1
        sequence_data[seq] = {
            "segs": kept,
            "coords": coords,
            "centroid": (float(np.median(lats)), float(np.median(lons))),
            "classes": dict(classes),
        }

    seg_to_sequence = {sid: seq for seq, entry in sequence_data.items() for sid in entry["segs"]}

    return sequence_data, seg_to_sequence


@st.cache_data
def load_class_names(metadata_dir: str, fingerprint: tuple) -> dict[tuple[str, int], str]:
    """Return ``{(attribute, code): "Attribute = Value"}`` for display.

    Read from ``attribute_metadata.json``, which is only needed to name classes.
    It is absent from some older metadata directories, so a missing or malformed
    file degrades the coverage report to ``"Attribute = code N"`` rather than
    disabling it: the classes themselves come from the segment data.
    ``fingerprint`` is unused; see :func:`load_unlabeled_sequence_data`.
    """
    path = Path(metadata_dir) / layout.ATTR_META_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        return {
            (attr, int(code)): f"{attr} = {value}"
            for attr, values in meta.get("attribute_value_to_irap_number", {}).items()
            for value, code in values.items()
        }
    except (ValueError, TypeError, AttributeError):
        return {}


# Class coverage


@dataclass
class ClassCoverage:
    """Per-split support for every (attribute, IRAP code) class in the data.

    Support is measured in distinct sequences (see :data:`DEFAULT_MIN_SEQ`).
    Classes are partitioned by whether the current assignment can be blamed for
    their shortfall:

    ``stratifiable``
        occur in at least ``STRATIFIABLE_FACTOR * n_min`` sequences, so some
        assignment covers them in every split. ``short`` lists those that the
        current one does not, and is the number worth acting on.
    ``irreducible``
        too rare for that to be possible at this ``n_min``, whatever the split.
    """

    n_min: int
    global_seqs: dict[tuple[str, int], int]
    global_segs: dict[tuple[str, int], int]
    split_seqs: dict[str, Counter]
    split_segs: dict[str, Counter]
    stratifiable: set[tuple[str, int]]
    absent: dict[str, list[tuple[str, int]]]
    short: dict[str, list[tuple[int, int, tuple[str, int]]]]
    irreducible: list[tuple[int, tuple[str, int]]]
    tvd: dict[str, dict[str, float]]

    @property
    def n_classes(self) -> int:
        return len(self.global_seqs)

    def n_absent_fixable(self, split: str) -> int:
        """Absent classes that some other assignment could have covered."""
        return sum(1 for c in self.absent[split] if c in self.stratifiable)

    def n_covered_everywhere(self) -> int:
        """Stratifiable classes already at ``n_min`` sequences in every split."""
        return sum(
            1
            for c in self.stratifiable
            if all(self.split_seqs[s][c] >= self.n_min for s in REAL_SPLITS)
        )

    def label(self, cls: tuple[str, int], names: dict[tuple[str, int], str]) -> str:
        attr, code = cls
        return names.get(cls, f"{attr} = code {code}")


def compute_class_coverage(
    sequence_data: dict[str, dict],
    assignments: dict[str, str],
    n_min: int,
) -> ClassCoverage:
    """Summarise class support per split for the current assignment.

    Runs on every rerun, i.e. after every box drawn, so it walks the
    pre-aggregated per-sequence histograms rather than the segments themselves –
    a few tens of thousands of increments, negligible next to the redraw.
    """
    global_seqs: Counter = Counter()
    global_segs: Counter = Counter()
    split_seqs: dict[str, Counter] = {s: Counter() for s in SPLITS}
    split_segs: dict[str, Counter] = {s: Counter() for s in SPLITS}

    for seq, entry in sequence_data.items():
        split = assignments.get(seq, "none")
        for cls, n in entry["classes"].items():
            global_seqs[cls] += 1
            global_segs[cls] += n
            split_seqs[split][cls] += 1
            split_segs[split][cls] += n

    stratifiable = {c for c, g in global_seqs.items() if g >= STRATIFIABLE_FACTOR * n_min}

    absent: dict[str, list] = {}
    short: dict[str, list] = {}
    for split in REAL_SPLITS:
        absent[split] = sorted(
            (c for c in global_seqs if split_segs[split][c] == 0),
            key=lambda c: (-global_seqs[c], c),
        )
        # Worst first: no sequences at all before merely thin, and among equals
        # the classes with most to draw from, which are the easiest to fix.
        short[split] = sorted(
            (
                (split_seqs[split][c], split_segs[split][c], c)
                for c in stratifiable
                if split_seqs[split][c] < n_min
            ),
            key=lambda t: (t[0], -global_seqs[t[2]], t[2]),
        )

    irreducible = sorted(
        ((global_seqs[c], c) for c in global_seqs if c not in stratifiable),
        key=lambda t: (-t[0], t[1]),
    )

    # Distribution match: how far each split's class mix has drifted from the
    # dataset as a whole, per attribute. Total variation distance reads directly
    # as the fraction of probability mass sitting in the wrong class.
    by_attr: dict[str, list] = defaultdict(list)
    for cls in global_segs:
        by_attr[cls[0]].append(cls)
    tvd: dict[str, dict[str, float]] = {}
    for split in REAL_SPLITS:
        per_attr: dict[str, float] = {}
        for attr, classes in by_attr.items():
            g_total = sum(global_segs[c] for c in classes)
            s_total = sum(split_segs[split][c] for c in classes)
            if not g_total or not s_total:
                continue
            per_attr[attr] = 0.5 * sum(
                abs(split_segs[split][c] / s_total - global_segs[c] / g_total)
                for c in classes
            )
        tvd[split] = per_attr

    return ClassCoverage(
        n_min=n_min,
        global_seqs=dict(global_seqs),
        global_segs=dict(global_segs),
        split_seqs=split_seqs,
        split_segs=split_segs,
        stratifiable=stratifiable,
        absent=absent,
        short=short,
        irreducible=irreducible,
        tvd=tvd,
    )


def _load_existing(
    output_path: Path,
    seg_to_sequence: dict[str, str],
    key_prefix: str = "",
) -> tuple[dict[str, str], int, int, dict[str, dict[str, int]]]:
    """Return (sequence → split, num_unmatched_seg_ids, num_seg_ids, conflicts).

    Only keys starting with ``key_prefix`` are read; the prefix is stripped
    before validation against :data:`SPLITS`. Seg_ids that resolve to no known
    sequence are counted rather than silently dropped, so a stale or mismatched
    splits.json is visible instead of quietly resetting sequences to "none".

    splits.json stores seg_ids while the editor works per sequence, so one
    sequence can pick up seg_ids from several splits. That happens whenever a
    rebuild changes sequence membership – most notably when new coding-table
    rows fix an adjacency violation and ``<section>__part0``/``__part1`` merge
    back into a single ``<section>``, each part carrying a different split.
    Such a sequence is resolved by majority of seg_ids (ties broken by
    :data:`SPLITS` order) and reported in ``conflicts`` as
    ``{sequence: {split: n_seg_ids}}``, rather than being decided silently by
    whichever key happened to come last in the file.

    Returns an empty mapping if the file does not exist.
    """
    if not output_path.exists():
        return {}, 0, 0, {}
    with open(output_path, encoding="utf-8") as f:
        raw: dict = json.load(f)
    votes: dict[str, dict[str, int]] = {}
    n_total = 0
    n_unmatched = 0
    for key, seg_ids in raw.items():
        if not key.startswith(key_prefix):
            continue
        split = key[len(key_prefix) :]
        # Rejects the "unlabeled_*" keys when key_prefix is "" (they are not
        # split names), and "unlabeled_unlocated" when it is "unlabeled_".
        if not split or split not in SPLITS:
            continue
        for sid in seg_ids:
            n_total += 1
            seq = seg_to_sequence.get(str(sid))
            if seq:
                counts = votes.setdefault(seq, {})
                counts[split] = counts.get(split, 0) + 1
            else:
                n_unmatched += 1
    result = {
        seq: max(counts, key=lambda s: (counts[s], -SPLITS.index(s)))
        for seq, counts in votes.items()
    }
    conflicts = {seq: counts for seq, counts in votes.items() if len(counts) > 1}
    return result, n_unmatched, n_total, conflicts


# Plotly figure


def _data_bbox(sequence_data: dict) -> tuple[float, float, float, float] | None:
    """Return (lat_min, lat_max, lon_min, lon_max) over all coords, or None."""
    lats: list[float] = []
    lons: list[float] = []
    for s in sequence_data.values():
        for lat, lon in s["coords"]:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        return None
    return min(lats), max(lats), min(lons), max(lons)


def _centroid_trace(layer: Layer, assignments: dict[str, str]) -> go.Scattergeo:
    """Build a single Scattergeo trace of all sequence centroids in ``layer``."""
    sequences = list(layer.data)
    lats = [layer.data[s]["centroid"][0] for s in sequences]
    lons = [layer.data[s]["centroid"][1] for s in sequences]
    colors = [layer.colors[assignments[s]] for s in sequences]
    tag = " (unlabeled)" if layer.name == "unlabeled" else ""
    hover = [f"{s}{tag}<br>{len(layer.data[s]['segs'])} segs · {layer.key_prefix}{assignments[s]}" for s in sequences]
    return go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=layer.marker_size, color=colors, symbol="circle"),
        customdata=[[s, layer.name] for s in sequences],
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name=f"{layer.name}_centroids",
        showlegend=False,
    )


def _build_figure(
    labeled: Layer,
    labeled_assignments: dict[str, str],
    unlabeled: Layer,
    unlabeled_assignments: dict[str, str],
) -> tuple[go.Figure, dict]:
    fig = go.Figure()

    # Layer 1: road polylines – 4 aggregated traces (one per split)
    for split, color in COLORS.items():
        lats: list = []
        lons: list = []
        for seq, sp in labeled_assignments.items():
            if sp != split:
                continue
            coords = labeled.data[seq]["coords"]
            lats += [c[0] for c in coords] + [None]
            lons += [c[1] for c in coords] + [None]
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(color=color, width=2),
                name=split,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Layer 2: unlabeled centroids (drawn under labeled so labeled stays on top)
    if unlabeled.data:
        fig.add_trace(_centroid_trace(unlabeled, unlabeled_assignments))

    # Layer 3: labeled centroids – used for box selection
    fig.add_trace(_centroid_trace(labeled, labeled_assignments))

    bbox = _data_bbox(labeled.data)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            projection=dict(type="mercator"),
            fitbounds="locations",
            visible=True,
            showcountries=True,
            countrycolor="rgba(0,0,0,0.5)",
            showcoastlines=True,
            coastlinecolor="rgba(0,0,0,0.6)",
            showland=True,
            landcolor="rgb(243,243,243)",
            showocean=True,
            oceancolor="rgb(220,235,245)",
            showframe=False,
            resolution=50,
        ),
        dragmode="select",
        modebar=dict(
            orientation="v",
            bgcolor="rgba(255,255,255,1)",
            color="rgba(0,0,0,1)",
            activecolor="rgba(20,80,160,1)",
        ),
        height=700,
        uirevision="stable",  # preserve zoom/pan across reruns
    )
    diag = {
        "n_sequences": len(labeled.data),
        "n_centroids": len(labeled.data),
        "bbox": bbox,
    }
    return fig, diag


def _sidebar_row(layer: Layer, assignments: dict[str, str], split: str, total_segs: int, indent: bool) -> str:
    """Return the HTML for one sidebar metric row."""
    color = layer.colors[split]
    n = sum(len(layer.data[s]["segs"]) for s, sp in assignments.items() if sp == split)
    pct = f" ({100 * n / total_segs:.1f}%)" if total_segs and split != "none" else ""
    seqs = sum(1 for sp in assignments.values() if sp == split)
    label = f"{layer.key_prefix}{split}"
    if indent:
        return (
            f'<div style="margin:0 0 6px 0;font-size:0.85rem;color:var(--sidebar-text-muted);">'
            f'<span style="background:{color};display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>'
            f"{label}: {n} segments{pct}, {seqs} sequences"
            f"</div>"
        )
    return (
        f'<div style="margin-bottom:2px;">'
        f'<span style="background:{color};display:inline-block;width:14px;height:14px;'
        f'border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
        f"<strong>{label}</strong>: {n} segments{pct}, {seqs} sequences"
        f"</div>"
    )


def _coverage_row_html(coverage: ClassCoverage, split: str) -> str:
    """Return the HTML for the class-coverage line under a split's metric row.

    Reports only how many classes have no segment at all in the split, and how
    many of those any assignment of whole sequences could have covered. The
    shortfall against ``n_min`` sequences lives in the detail panel instead.

    Colouring follows the fixable count, not the raw one: a split missing only
    classes that are too rare to place everywhere has nothing to act on, and
    flagging it would be a false alarm that never clears.
    """
    n_absent = len(coverage.absent[split])
    n_fixable = coverage.n_absent_fixable(split)
    color = "var(--sidebar-text-dim)" if n_fixable else "var(--sidebar-text)"
    noun = "class" if n_absent == 1 else "classes"
    text = f"{n_absent} {noun} absent ({n_fixable} fixable)" if n_absent else "no class absent"
    return (
        f'<div style="margin:0 0 6px 22px;font-size:0.8rem;color:{color};">'
        f"{text}"
        f"</div>"
    )


def _class_list_html(
    entries: list[str],
    limit: int = 12,
) -> str:
    """Return a compact, scannable list of class descriptions.

    Entries are escaped: IRAP value names routinely contain ``<`` and ``>``
    ("Grade = 0% to <2.5%", "Residential access <3"), which the sidebar would
    otherwise render as the start of a tag and swallow the rest of the line.
    """
    shown = entries[:limit]
    more = (
        f'<div style="color:var(--sidebar-text-dim);">… and {len(entries) - limit} more</div>'
        if len(entries) > limit
        else ""
    )
    body = "".join(f'<div style="margin-bottom:2px;">{html.escape(e)}</div>' for e in shown)
    return (
        f'<div style="font-size:0.75rem;line-height:1.35;color:var(--sidebar-text);'
        f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">'
        f"{body}{more}</div>"
    )


def _render_coverage_panel(coverage: ClassCoverage, names: dict[tuple[str, int], str]) -> None:
    """Render the expandable class-coverage detail panel in the sidebar."""
    n_ok = coverage.n_covered_everywhere()
    n_strat = len(coverage.stratifiable)
    with st.sidebar.expander(f"Class coverage — {n_ok}/{n_strat} classes in every split"):
        st.number_input(
            "Min sequences per class per split",
            min_value=1,
            max_value=20,
            value=DEFAULT_MIN_SEQ,
            step=1,
            key="class_n_min",
            help=(
                "Support is counted in distinct sequences, not segments: "
                "consecutive segments of one road carry near-identical "
                "attributes, so many segments from a single section are one "
                "example, not many."
            ),
        )
        st.caption(
            f"{coverage.n_classes} (attribute, class) pairs occur in the data. "
            f"{n_strat} of them appear in at least "
            f"{STRATIFIABLE_FACTOR * coverage.n_min} sequences, so some assignment "
            f"covers them everywhere; the rest are listed at the bottom as "
            f"unfixable. TVD is the largest per-attribute total variation "
            f"distance from the dataset-wide class mix."
        )

        header = (
            '<tr style="text-align:left;color:var(--sidebar-text);">'
            "<th>split</th><th>absent</th><th>fixable</th><th>short</th><th>TVD</th></tr>"
        )
        rows = ""
        for split in REAL_SPLITS:
            # An unassigned split has no distribution to compare, and printing
            # 0.000 there would read as a perfect match rather than "no data".
            worst_tvd = f"{max(coverage.tvd[split].values()):.3f}" if coverage.tvd[split] else "–"
            rows += (
                f'<tr><td style="color:{COLORS[split]};font-weight:600;">{split}</td>'
                f"<td>{len(coverage.absent[split])}</td>"
                f"<td>{coverage.n_absent_fixable(split)}</td>"
                f"<td>{len(coverage.short[split])}</td>"
                f"<td>{worst_tvd}</td></tr>"
            )
        st.markdown(
            f'<table style="font-size:0.78rem;width:100%;">{header}{rows}</table>',
            unsafe_allow_html=True,
        )

        for split in REAL_SPLITS:
            entries = [
                f"{n_seq} seq / {n_seg} seg — {coverage.label(cls, names)} "
                f"[{coverage.global_seqs[cls]} seq total]"
                for n_seq, n_seg, cls in coverage.short[split]
            ]
            st.markdown(
                f'<div style="margin:10px 0 4px 0;font-weight:600;color:{COLORS[split]};">'
                f"{split}: {len(entries)} short</div>",
                unsafe_allow_html=True,
            )
            if entries:
                st.markdown(_class_list_html(entries), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:var(--sidebar-text);">'
                    f"every stratifiable class has ≥{coverage.n_min} sequences</div>",
                    unsafe_allow_html=True,
                )

        if coverage.irreducible:
            st.markdown(
                f'<div style="margin:12px 0 4px 0;font-weight:600;color:var(--sidebar-text);">'
                f"{len(coverage.irreducible)} classes cannot be stratified</div>"
                f'<div style="font-size:0.75rem;color:var(--sidebar-text-dim);margin-bottom:4px;">'
                f"fewer than {STRATIFIABLE_FACTOR * coverage.n_min} sequences exist "
                f"dataset-wide; no assignment covers these in every split</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _class_list_html(
                    [
                        f"{n} seq — {coverage.label(cls, names)}"
                        for n, cls in coverage.irreducible
                    ]
                ),
                unsafe_allow_html=True,
            )


# App


def main() -> None:
    st.set_page_config(page_title="Split Editor", layout="wide")
    args = _parse_args()

    fingerprint = _metadata_fingerprint(args.metadata_dir)
    sequence_data, seg_to_sequence = load_sequence_data(str(args.metadata_dir), fingerprint)
    unlabeled_sequence_data: dict[str, dict] = load_unlabeled_sequence_data(
        str(args.metadata_dir), fingerprint
    )
    unlocated_seg_ids: list[str] = load_unlocated_seg_ids(str(args.metadata_dir), fingerprint)
    class_names = load_class_names(str(args.metadata_dir), fingerprint)

    labeled = Layer(
        name="labeled",
        data=sequence_data,
        colors=COLORS,
        marker_size=8,
        key_prefix="",
        state_key="sequence_to_split",
    )
    unlabeled = Layer(
        name="unlabeled",
        data=unlabeled_sequence_data,
        colors=UNLABELED_COLORS,
        marker_size=6,
        key_prefix="unlabeled_",
        state_key="unlabeled_sequence_to_split",
    )
    layers: tuple[Layer, ...] = (labeled, unlabeled)

    # A stale unlabeled_unlocated_segment_ids.json – written by an earlier build
    # and left behind by a later one that placed those image folders into real
    # sequences – would otherwise be echoed back on save, putting the same
    # seg_id under both "unlabeled_<split>" and "unlabeled_unlocated".
    known_seg_ids = {
        str(sid) for layer in layers for entry in layer.data.values() for sid in entry["segs"]
    }
    n_stale_unlocated = sum(1 for s in unlocated_seg_ids if s in known_seg_ids)
    n_unlocated_on_disk = len(unlocated_seg_ids)
    unlocated_seg_ids = [s for s in unlocated_seg_ids if s not in known_seg_ids]

    # Session state init. Keyed on the metadata fingerprint rather than mere
    # presence, so that re-running build_metadata.py under a live session
    # rebuilds the assignments instead of leaving them keyed by sequences that
    # no longer exist.
    if st.session_state.get("metadata_fingerprint") != fingerprint:
        st.session_state.metadata_fingerprint = fingerprint
        u_seg_to_sequence = {str(sid): seq for seq, entry in unlabeled_sequence_data.items() for sid in entry["segs"]}
        seg_lookup = {labeled.name: seg_to_sequence, unlabeled.name: u_seg_to_sequence}
        warnings: list[str] = []
        for layer in layers:
            existing, n_unmatched, n_total, conflicts = _load_existing(
                args.output, seg_lookup[layer.name], layer.key_prefix
            )
            st.session_state[layer.state_key] = {s: existing.get(s, "none") for s in layer.data}
            if n_unmatched:
                warnings.append(
                    f"splits.json: {n_unmatched} of {n_total} {layer.name} seg_ids matched no "
                    f"sequence and were dropped – the file predates the current metadata."
                )
            if conflicts:
                shown = "; ".join(
                    f"{seq} → {existing[seq]} "
                    f"({', '.join(f'{n}×{sp}' for sp, n in sorted(counts.items()))})"
                    for seq, counts in sorted(conflicts.items())[:5]
                )
                more = f"; and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
                warnings.append(
                    f"splits.json: {len(conflicts)} {layer.name} sequence(s) carry seg_ids from "
                    f"more than one split, so a rebuild changed their membership. Resolved by "
                    f"majority of seg_ids: {shown}{more}. Re-check these on the map before saving."
                )
        if n_stale_unlocated:
            warnings.append(
                f"{layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME} is stale: "
                f"{n_stale_unlocated} of its {n_unlocated_on_disk} seg_ids now belong to a real "
                f"sequence. They are excluded from unlabeled_unlocated so no seg_id lands in two "
                f"splits; re-run build_metadata.py to refresh the file."
            )
        st.session_state.load_warnings = warnings
        st.session_state.history: list[dict] = []

    assignments = {layer.name: st.session_state[layer.state_key] for layer in layers}

    # Sidebar
    st.sidebar.title("Split Editor")

    for warning in st.session_state.get("load_warnings", []):
        st.sidebar.warning(warning)

    active_split = st.sidebar.radio(
        "Active split (draw a box to assign)",
        SPLITS,
        horizontal=False,
    )

    st.sidebar.markdown(
        """
        <style>
        :root {
            --sidebar-text: var(--text-color);
            --sidebar-text-muted: var(--text-color-muted);
            --sidebar-text-dim: var(--text-color-disabled);
        }
        [data-testid="stSidebarHeader"] {
            display: none;
        }
        [data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stAppViewContainer"] > .main {
            top: 0;
        }
        [data-testid="stMainBlockContainer"] {
            padding: 0rem;
            max-width: 100%;
            overflow: hidden;
        }
        .modebar-container {
            top: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
            font-size: 0.8rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.0rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Read before the widget that sets it is created further down: Streamlit
    # reruns the whole script on a widget change, so session_state already holds
    # the new value here and the rows below stay consistent with the panel.
    n_min = int(st.session_state.get("class_n_min", DEFAULT_MIN_SEQ))
    coverage = compute_class_coverage(labeled.data, assignments[labeled.name], n_min)

    totals = {layer.name: sum(len(layer.data[s]["segs"]) for s in layer.data) for layer in layers}
    for split in SPLITS:
        st.sidebar.markdown(
            _sidebar_row(labeled, assignments[labeled.name], split, totals[labeled.name], indent=False),
            unsafe_allow_html=True,
        )
        if split in REAL_SPLITS and coverage.n_classes:
            st.sidebar.markdown(_coverage_row_html(coverage, split), unsafe_allow_html=True)
        if unlabeled.data:
            st.sidebar.markdown(
                _sidebar_row(unlabeled, assignments[unlabeled.name], split, totals[unlabeled.name], indent=True),
                unsafe_allow_html=True,
            )

    if unlocated_seg_ids:
        unlocated_color = UNLABELED_COLORS["none"]
        st.sidebar.markdown(
            f'<div style="margin:0 0 6px 0;font-size:0.85rem;color:var(--sidebar-text-muted);">'
            f'<span style="background:{unlocated_color};display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>'
            f"unlabeled_unlocated: {len(unlocated_seg_ids)} segments "
            f"(auto-assigned on save)"
            f"</div>",
            unsafe_allow_html=True,
        )

    if coverage.n_classes:
        _render_coverage_panel(coverage, class_names)

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    undo_clicked = col1.button("Undo", width="stretch", disabled=not st.session_state.history)
    reset_clicked = col2.button("Reset", width="stretch")

    st.sidebar.markdown("")
    save_clicked = st.sidebar.button("Save splits.json", width="stretch", type="primary")

    def _snapshot() -> dict:
        return {layer.name: dict(assignments[layer.name]) for layer in layers}

    # Handle sidebar actions before rendering the map
    if undo_clicked and st.session_state.history:
        snapshot = st.session_state.history.pop()
        for layer in layers:
            st.session_state[layer.state_key] = snapshot[layer.name]
        st.rerun()

    if reset_clicked:
        st.session_state.history.append(_snapshot())
        for layer in layers:
            st.session_state[layer.state_key] = {s: "none" for s in layer.data}
        st.rerun()

    if save_clicked:
        out: dict[str, list] = {s: [] for s in SPLITS if s != "none"}
        for layer in layers:
            for seq, sp in assignments[layer.name].items():
                if sp == "none":
                    continue
                key = f"{layer.key_prefix}{sp}"
                out.setdefault(key, []).extend(layer.data[seq]["segs"])
        if unlocated_seg_ids:
            out["unlabeled_unlocated"] = list(unlocated_seg_ids)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        # The file now matches the in-session state, so any load warning is stale.
        st.session_state.load_warnings = []
        st.sidebar.success(f"Saved {args.output}")

    # Map
    fig, diag = _build_figure(
        labeled,
        assignments[labeled.name],
        unlabeled,
        assignments[unlabeled.name],
    )
    event = st.plotly_chart(
        fig,
        on_select="rerun",
        width="stretch",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [],
            "showTips": False,
            "responsive": True,
        },
    )

    # Handle map selection
    if event and event.selection and event.selection.points:
        prev = _snapshot()
        target_by_kind = {layer.name: assignments[layer.name] for layer in layers}
        changed = False
        for p in event.selection.points:
            cd = p.get("customdata")
            if not cd:
                continue
            seq, kind = cd[0], cd[1]
            target = target_by_kind.get(kind)
            if target is not None and seq in target and target[seq] != active_split:
                target[seq] = active_split
                changed = True
        if changed:
            st.session_state.history.append(prev)
            st.rerun()


if __name__ == "__main__":
    main()
