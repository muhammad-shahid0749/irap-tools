"""Interactive map GUI for manually assigning sequences to train/val/test splits.

Requires Streamlit >= 1.52 (for ``width="stretch"`` of ``st.plotly_chart``).

Launch:
    streamlit run split_editor.py -- <data_dir>

Reads the metadata files that build_metadata.py writes to ``<data_dir>``
(:data:`METADATA_PATH_FUNCTIONS`) and writes ``<data_dir>/splits.json``.

Interaction:
    1. Pick an active split in the sidebar (train / val / test / none).
    2. Draw a rectangle on the map – all sequences whose centroid falls
       inside are assigned to the active split.
    3. Undo reverts the last assignment batch. Reset clears all.
    4. Save writes splits.json (format documented in README.md, "`splits.json` format").

The sidebar reports class coverage per split: how many (attribute, iRAP code)
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
import dataclasses as dc
import html
import json
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import layout
from parse_coding_tables import is_real_attribute_code


# Splits ###########################################################################################

REAL_SPLITS = ("train", "val", "test")
UNASSIGNED = "none"
SPLITS = (*REAL_SPLITS, UNASSIGNED)

# Keys of splits.json other than the names of REAL_SPLITS.
UNLABELED_SPLIT_PREFIX = "unlabeled_"
UNLOCATED_SPLIT = "unlabeled_unlocated"

COLORS: dict[str, str] = {
    "train": "#4477AA",
    "val": "#CC3333",
    "test": "#559933",
    UNASSIGNED: "#BBBBBB",
}


def _muted(hex_color: str, t: float = 0.5) -> str:
    """Interpolates ``hex_color`` toward grey ``#BBBBBB`` by factor ``t``."""
    grey = 0xBB
    channels = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c * (1 - t) + grey * t):02X}" for c in channels)


UNLABELED_COLORS: dict[str, str] = {s: _muted(c) for s, c in COLORS.items()}

# Layer name -> sequence -> split. The editor replaces these dictionaries instead of changing them,
# so the undo history can hold them without copies.
Assignments = dict[str, dict[str, str]]


@dc.dataclass(frozen=True)
class Layer:
    """A category of sequences (labeled or unlabeled) rendered as one centroid trace."""

    name: str  # "labeled" | "unlabeled"
    data: dict[str, dict]  # sequence -> {"segs": [seg_id, ...], "centroid": (lat, lon), ...}
    colors: dict[str, str]
    marker_size: int
    split_key_prefix: str

    def split_key(self, split: str) -> str:
        """Returns the splits.json key of the layer's sequences in ``split``."""
        return f"{self.split_key_prefix}{split}"


# Class-coverage settings ##########################################################################

# A "class" is one (attribute, iRAP code) pair.
AttributeClass = tuple[str, int]

# Support is counted in distinct *sequences*, not segments: consecutive 20 m
# segments of one road carry near-identical attributes, so 500 segments of a
# rare class drawn from a single section are one example, not 500.
DEFAULT_MIN_NUM_SEQUENCES = 3
MIN_NUM_SEQUENCES_KEY = "min_num_sequences"  # in st.session_state

# A class needs the minimum number of sequences in each of the three splits, so
# unless it occurs in at least this many times the minimum dataset-wide, no
# assignment of whole sequences can cover it everywhere. Such classes are
# reported separately rather than counted against the current assignment, which
# would otherwise show a floor that no amount of redrawing can bring down.
STRATIFIABLE_FACTOR = len(REAL_SPLITS)


# Metadata loading #################################################################################

# The files written by build_metadata.py that the editor reads.
METADATA_PATH_FUNCTIONS = (
    layout.segment_id_to_road_data_path,
    layout.road_id_to_segment_id_sequence_path,
    layout.unlabeled_sequence_id_to_data_path,
    layout.unlabeled_unlocated_segment_ids_path,
    layout.output_attr_meta_path,
)


@dc.dataclass(frozen=True)
class EditorMetadata:
    sequence_data: dict[str, dict]  # see `load_sequence_data`
    unlabeled_sequence_data: dict[str, dict]  # see `load_unlabeled_sequence_data`
    # Without the seg_ids that are in a sequence. A stale
    # unlabeled_unlocated_segment_ids.json – written by an earlier build and
    # left behind by a later one that placed those image folders into real
    # sequences – would otherwise put the same seg_id under both
    # "unlabeled_<split>" and "unlabeled_unlocated" on save.
    unlocated_seg_ids: list[str]
    num_stale_unlocated_seg_ids: int
    class_names: dict[AttributeClass, str]  # see `load_class_names`


def _load_json(path: Path) -> T.Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_metadata_fingerprint(data_dir: Path) -> tuple:
    """Returns the (name, mtime, size) of each file of :data:`METADATA_PATH_FUNCTIONS`.

    It is the cache key of :func:`load_metadata` and of the assignments in the
    session state, so that both are rebuilt after ``build_metadata.py`` was
    re-run. Otherwise a Save would write splits.json from obsolete sequence
    membership.
    """
    fingerprint = []
    for path_f in METADATA_PATH_FUNCTIONS:
        path = path_f(data_dir)
        try:
            stat = path.stat()
        except OSError:
            fingerprint.append((path.name, None, None))
        else:
            fingerprint.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


def load_sequence_data(data_dir: Path) -> dict[str, dict]:
    """Loads the labeled sequences.

    Returns:
        sequence -> {
            "segs": [seg_id, ...],          # ordered
            "coords": [(lat, lon), ...],    # parallel to segs
            "centroid": (lat, lon),
            "classes": {(attribute, code): num_segments, ...},
        }

    Sequences are the road_ids of ``road_id_to_segment_id_sequence.json``, *not*
    the ``section`` field of ``segment_id_to_road_data.json``: a section whose
    seg_id/distance adjacency is inconsistent is emitted as several
    ``"<section>__partN"`` road_ids. Only the seg_ids with road data are kept.

    ``classes`` is the sequence's histogram over the ``required_attributes``
    of its segments, and is what the class coverage report is summed from.
    Non-genuine codes (``MISSING_ATTR_CODE`` and ``None``) are excluded, so an
    attribute that was never coded contributes no classes at all rather than a
    spurious "missing" one.
    """
    seg_to_road: dict = _load_json(layout.segment_id_to_road_data_path(data_dir))
    road_to_segs: dict = _load_json(layout.road_id_to_segment_id_sequence_path(data_dir))

    sequence_data: dict = {}
    for seq, segs in road_to_segs.items():
        kept = [s for s in segs if s in seg_to_road]
        if not kept:
            continue
        coords = [(seg_to_road[sid]["lat"], seg_to_road[sid]["lon"]) for sid in kept]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        classes: Counter = Counter()
        for sid in kept:
            for attr, code in seg_to_road[sid].get("required_attributes", {}).items():
                if is_real_attribute_code(code):
                    classes[(attr, int(code))] += 1
        sequence_data[seq] = {
            "segs": kept,
            "coords": coords,
            "centroid": (float(np.median(lats)), float(np.median(lons))),
            "classes": dict(classes),
        }
    return sequence_data


def load_unlabeled_sequence_data(data_dir: Path) -> dict[str, dict]:
    """Loads the unlabeled sequences, ``{}`` without the file.

    Returns:
        sequence -> {"segs": [seg_id, ...], "centroid": (lat, lon)}
    """
    path = layout.unlabeled_sequence_id_to_data_path(data_dir)
    if not path.exists():
        return {}
    return {
        seq: {
            "segs": list(entry["segs"]),
            "centroid": (float(entry["centroid"][0]), float(entry["centroid"][1])),
        }
        for seq, entry in _load_json(path).items()
    }


def load_unlocated_seg_ids(data_dir: Path) -> list[str]:
    """Loads the unlabeled seg_ids with no derivable map location, ``[]`` without the file.

    These come from image folders that have no labeled siblings. They are
    auto-assigned to the :data:`UNLOCATED_SPLIT` split.
    """
    path = layout.unlabeled_unlocated_segment_ids_path(data_dir)
    if not path.exists():
        return []
    return [str(s) for s in _load_json(path)]


def load_class_names(data_dir: Path) -> dict[AttributeClass, str]:
    """Loads ``{(attribute, code): "Attribute = Value"}`` for display.

    Read from ``attribute_metadata.json``, which is only needed to name classes.
    It is absent from some older metadata directories, so a missing or malformed
    file degrades the coverage report to ``"Attribute = code N"`` rather than
    disabling it: the classes themselves come from the segment data.
    """
    path = layout.output_attr_meta_path(data_dir)
    if not path.exists():
        return {}
    try:
        return {
            (attr, int(code)): f"{attr} = {value}"
            for attr, values in _load_json(path).get("attribute_value_to_irap_number", {}).items()
            for value, code in values.items()
        }
    except (ValueError, TypeError, AttributeError):
        return {}


def invert_sequence_segments(sequence_data: dict[str, dict]) -> dict[str, str]:
    """Returns seg_id -> sequence."""
    return {seg_id: seq for seq, entry in sequence_data.items() for seg_id in entry["segs"]}


@st.cache_data
def load_metadata(data_dir: Path, fingerprint: tuple) -> EditorMetadata:
    """Loads the files of :data:`METADATA_PATH_FUNCTIONS`.

    ``st.cache_data`` caches for the whole server process and keys on the
    argument values. ``fingerprint`` (:func:`compute_metadata_fingerprint`) is
    only a part of that key.
    """
    sequence_data = load_sequence_data(data_dir)
    unlabeled_sequence_data = load_unlabeled_sequence_data(data_dir)
    unlocated_seg_ids_in_file = load_unlocated_seg_ids(data_dir)
    located_seg_ids = (invert_sequence_segments(sequence_data).keys()
                       | invert_sequence_segments(unlabeled_sequence_data).keys())
    unlocated_seg_ids = [s for s in unlocated_seg_ids_in_file if s not in located_seg_ids]
    return EditorMetadata(
        sequence_data=sequence_data,
        unlabeled_sequence_data=unlabeled_sequence_data,
        unlocated_seg_ids=unlocated_seg_ids,
        num_stale_unlocated_seg_ids=len(unlocated_seg_ids_in_file) - len(unlocated_seg_ids),
        class_names=load_class_names(data_dir),
    )


# Class coverage ###################################################################################


@dc.dataclass(frozen=True)
class ClassCoverage:
    """Per-split support for every (attribute, iRAP code) class in the data.

    Support is measured in distinct sequences (see :data:`DEFAULT_MIN_NUM_SEQUENCES`).
    Classes are partitioned by whether the current assignment can be blamed for
    their shortfall:

    ``stratifiable``
        occur in at least ``min_num_sequences_to_stratify`` sequences, so some
        assignment covers them in every split. ``short`` lists those that the
        current one does not, and is the number worth acting on.
    ``irreducible``
        too rare for that to be possible at this ``min_num_sequences``, whatever
        the split.
    """

    min_num_sequences: int
    class_to_num_sequences: dict[AttributeClass, int]  # dataset-wide
    stratifiable: set[AttributeClass]
    # Split -> the classes with no segment in the split.
    absent: dict[str, set[AttributeClass]]
    # Split -> (number of sequences, number of segments, class) of the stratifiable classes with
    # fewer than `min_num_sequences` sequences in the split, worst first.
    short: dict[str, list[tuple[int, int, AttributeClass]]]
    # (dataset-wide number of sequences, class), most frequent first.
    irreducible: list[tuple[int, AttributeClass]]
    # Split -> the largest per-attribute total variation distance between the class distributions
    # (over segments) of the split and of the dataset, None for a split without segments.
    max_tvd: dict[str, float | None]

    @property
    def min_num_sequences_to_stratify(self) -> int:
        return STRATIFIABLE_FACTOR * self.min_num_sequences

    @property
    def num_classes(self) -> int:
        return len(self.class_to_num_sequences)

    def num_absent_fixable(self, split: str) -> int:
        """Returns the number of absent classes that another assignment could cover."""
        return len(self.absent[split] & self.stratifiable)

    @property
    def num_covered_everywhere(self) -> int:
        """The number of stratifiable classes that no split has too few sequences of."""
        short_somewhere = {cls for entries in self.short.values() for _, _, cls in entries}
        return len(self.stratifiable) - len(short_somewhere)


def get_class_name(cls: AttributeClass, class_names: dict[AttributeClass, str]) -> str:
    attr, code = cls
    return class_names.get(cls, f"{attr} = code {code}")


def compute_class_coverage(
    sequence_data: dict[str, dict],
    sequence_to_split: dict[str, str],
    min_num_sequences: int,
) -> ClassCoverage:
    """Summarises the class support of each split under the assignment ``sequence_to_split``.

    It runs after every box drawn, so it sums the per-sequence histograms of
    :func:`load_sequence_data` instead of visiting the segments.
    """
    num_sequences: Counter = Counter()
    num_segments: Counter = Counter()
    split_num_sequences: dict[str, Counter] = {s: Counter() for s in SPLITS}
    split_num_segments: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for seq, entry in sequence_data.items():
        split = sequence_to_split[seq]
        for cls, n in entry["classes"].items():
            num_sequences[cls] += 1
            num_segments[cls] += n
            split_num_sequences[split][cls] += 1
            split_num_segments[split][cls] += n

    stratifiable = {c for c, n in num_sequences.items()
                    if n >= STRATIFIABLE_FACTOR * min_num_sequences}

    absent: dict[str, set] = {}
    short: dict[str, list] = {}
    for split in REAL_SPLITS:
        absent[split] = {c for c in num_sequences if split_num_sequences[split][c] == 0}
        # Worst first: no sequences at all before merely thin, and among equals
        # the classes with most to draw from, which are the easiest to fix.
        short[split] = sorted(
            (
                (split_num_sequences[split][c], split_num_segments[split][c], c)
                for c in stratifiable
                if split_num_sequences[split][c] < min_num_sequences
            ),
            key=lambda t: (t[0], -num_sequences[t[2]], t[2]),
        )

    irreducible = sorted(
        ((n, c) for c, n in num_sequences.items() if c not in stratifiable),
        key=lambda t: (-t[0], t[1]),
    )

    # Distribution match: how far each split's class mix has drifted from the
    # dataset as a whole, per attribute. Total variation distance reads directly
    # as the fraction of probability mass sitting in the wrong class.
    attr_to_classes: dict[str, list] = defaultdict(list)
    for cls in num_segments:
        attr_to_classes[cls[0]].append(cls)
    max_tvd: dict[str, float | None] = {}
    for split in REAL_SPLITS:
        tvds = []
        for classes in attr_to_classes.values():
            total = sum(num_segments[c] for c in classes)
            split_total = sum(split_num_segments[split][c] for c in classes)
            if split_total:
                tvds.append(0.5 * sum(
                    abs(split_num_segments[split][c] / split_total - num_segments[c] / total)
                    for c in classes
                ))
        max_tvd[split] = max(tvds, default=None)

    return ClassCoverage(
        min_num_sequences=min_num_sequences,
        class_to_num_sequences=dict(num_sequences),
        stratifiable=stratifiable,
        absent=absent,
        short=short,
        irreducible=irreducible,
        max_tvd=max_tvd,
    )


# Assignments and splits.json ######################################################################


@dc.dataclass(frozen=True)
class SplitsAssignment:
    """The assignment of sequences that the seg_ids of a splits.json give."""

    sequence_to_split: dict[str, str]  # only the sequences with a seg_id in a split
    num_seg_ids: int
    num_unmatched_seg_ids: int  # seg_ids in no sequence
    conflicts: dict[str, Counter]  # sequence -> split -> number of seg_ids, for > 1 split


def assign_sequences_from_splits(
    split_to_seg_ids: dict[str, list],
    seg_id_to_sequence: dict[str, str],
) -> SplitsAssignment:
    """Assigns each sequence to the split that holds most of its seg_ids.

    splits.json stores seg_ids while the editor works per sequence, so one
    sequence can pick up seg_ids from several splits. That happens whenever a
    rebuild changes sequence membership – most notably when new coding-table
    rows fix an adjacency violation and ``<section>__part0``/``__part1`` merge
    back into a single ``<section>``, each part carrying a different split.
    Such a sequence is resolved by majority of seg_ids (ties broken by
    :data:`SPLITS` order) and reported in ``conflicts``, rather than being
    decided by whichever key comes last in the file.

    Seg_ids that are in no sequence are counted rather than silently dropped, so
    a stale or mismatched splits.json is visible instead of quietly resetting
    sequences to :data:`UNASSIGNED`.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    num_seg_ids = 0
    num_unmatched_seg_ids = 0
    for split, seg_ids in split_to_seg_ids.items():
        for seg_id in seg_ids:
            num_seg_ids += 1
            seq = seg_id_to_sequence.get(str(seg_id))
            if seq is None:
                num_unmatched_seg_ids += 1
            else:
                votes[seq][split] += 1
    return SplitsAssignment(
        sequence_to_split={
            seq: max(counts, key=lambda s: (counts[s], -SPLITS.index(s)))
            for seq, counts in votes.items()
        },
        num_seg_ids=num_seg_ids,
        num_unmatched_seg_ids=num_unmatched_seg_ids,
        conflicts={seq: counts for seq, counts in votes.items() if len(counts) > 1},
    )


def format_splits_warnings(layer_name: str, assignment: SplitsAssignment) -> list[str]:
    """Describes how the splits.json that gave ``assignment`` differs from the metadata."""
    warnings = []
    if assignment.num_unmatched_seg_ids:
        warnings.append(
            f"{layout.SPLITS_FILENAME}: {assignment.num_unmatched_seg_ids} of "
            f"{assignment.num_seg_ids} {layer_name} seg_ids matched no "
            f"sequence and were dropped – the file predates the current metadata."
        )
    conflicts = assignment.conflicts
    if conflicts:
        max_num_shown = 5
        shown = "; ".join(
            f"{seq} → {assignment.sequence_to_split[seq]} "
            f"({', '.join(f'{n}×{split}' for split, n in sorted(counts.items()))})"
            for seq, counts in sorted(conflicts.items())[:max_num_shown]
        )
        more = (f"; and {len(conflicts) - max_num_shown} more"
                if len(conflicts) > max_num_shown else "")
        warnings.append(
            f"{layout.SPLITS_FILENAME}: {len(conflicts)} {layer_name} sequence(s) carry seg_ids "
            f"from more than one split, so a rebuild changed their membership. Resolved by "
            f"majority of seg_ids: {shown}{more}. Re-check these on the map before saving."
        )
    return warnings


def make_initial_assignments(
    splits: dict[str, list],
    layers: T.Sequence[Layer],
) -> tuple[Assignments, list[str]]:
    """Makes the starting assignments from the content of a splits.json, ``{}`` for no file.

    Returns:
        The assignments, with the sequences that ``splits`` does not assign in
        :data:`UNASSIGNED`, and warnings (:func:`format_splits_warnings`).
    """
    assignments: Assignments = {}
    warnings: list[str] = []
    for layer in layers:
        assignment = assign_sequences_from_splits(
            {split: splits.get(layer.split_key(split), []) for split in REAL_SPLITS},
            invert_sequence_segments(layer.data),
        )
        assignments[layer.name] = {
            seq: assignment.sequence_to_split.get(seq, UNASSIGNED) for seq in layer.data
        }
        warnings += format_splits_warnings(layer.name, assignment)
    return assignments, warnings


def make_splits(
    layers: T.Sequence[Layer],
    assignments: Assignments,
    unlocated_seg_ids: list[str],
) -> dict[str, list[str]]:
    """Makes the content of splits.json (format documented in README.md, "`splits.json` format")."""
    splits: dict[str, list[str]] = {split: [] for split in REAL_SPLITS}
    for layer in layers:
        for seq, split in assignments[layer.name].items():
            if split != UNASSIGNED:
                splits.setdefault(layer.split_key(split), []).extend(layer.data[seq]["segs"])
    if unlocated_seg_ids:
        splits[UNLOCATED_SPLIT] = list(unlocated_seg_ids)
    return splits


def assign_selected_sequences(
    assignments: Assignments,
    selected_points: list[dict],
    split: str,
) -> Assignments | None:
    """Assigns the sequences of the selected centroids (see :func:`_centroid_trace`) to ``split``.

    Returns:
        The new assignments, or None if the selection changes nothing.
    """
    new_assignments = {name: dict(seq_to_split) for name, seq_to_split in assignments.items()}
    selection_changes_assignments = False
    for point in selected_points:
        customdata = point.get("customdata")
        if not customdata:
            continue
        seq, layer_name = customdata
        seq_to_split = new_assignments.get(layer_name)
        if seq_to_split is not None and seq in seq_to_split and seq_to_split[seq] != split:
            seq_to_split[seq] = split
            selection_changes_assignments = True
    return new_assignments if selection_changes_assignments else None


# Plotly figure ####################################################################################


def _centroid_trace(layer: Layer, sequence_to_split: dict[str, str]) -> go.Scattergeo:
    """Builds a single Scattergeo trace of all sequence centroids in ``layer``.

    The ``customdata`` of a point is ``[sequence, layer name]``.
    """
    sequences = list(layer.data)
    tag = " (unlabeled)" if layer.name == "unlabeled" else ""
    return go.Scattergeo(
        lat=[layer.data[s]["centroid"][0] for s in sequences],
        lon=[layer.data[s]["centroid"][1] for s in sequences],
        mode="markers",
        marker=dict(
            size=layer.marker_size,
            color=[layer.colors[sequence_to_split[s]] for s in sequences],
        ),
        customdata=[[s, layer.name] for s in sequences],
        text=[
            f"{s}{tag}<br>{len(layer.data[s]['segs'])} segs · "
            f"{layer.split_key(sequence_to_split[s])}"
            for s in sequences
        ],
        hovertemplate="%{text}<extra></extra>",
        name=f"{layer.name}_centroids",
        showlegend=False,
    )


def _build_figure(labeled: Layer, unlabeled: Layer, assignments: Assignments) -> go.Figure:
    fig = go.Figure()

    # Layer 1: road polylines – one aggregated trace per split
    for split, color in COLORS.items():
        lats: list = []
        lons: list = []
        for seq, seq_split in assignments[labeled.name].items():
            if seq_split != split:
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
        fig.add_trace(_centroid_trace(unlabeled, assignments[unlabeled.name]))

    # Layer 3: labeled centroids
    fig.add_trace(_centroid_trace(labeled, assignments[labeled.name]))

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
    return fig


# Sidebar ##########################################################################################

PAGE_CSS = """
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
    </style>
    """


def _swatch_row(color: str, text: str, *, use_small_style: bool) -> str:
    """Returns the HTML for one sidebar row: a colour swatch followed by ``text``."""
    row_style, size_px, radius_px = (
        ("margin:0 0 6px 0;font-size:0.85rem;color:var(--sidebar-text-muted);", 10, 2)
        if use_small_style
        else ("margin-bottom:2px;", 14, 3)
    )
    return (
        f'<div style="{row_style}">'
        f'<span style="background:{color};display:inline-block;width:{size_px}px;'
        f'height:{size_px}px;border-radius:{radius_px}px;margin-right:6px;'
        f'vertical-align:middle;"></span>'
        f"{text}</div>"
    )


def _sidebar_row(
    layer: Layer, sequence_to_split: dict[str, str], split: str, *, use_small_style: bool
) -> str:
    """Returns the HTML for the sidebar row with the size of ``split`` in ``layer``."""
    num_segments = sum(
        len(layer.data[seq]["segs"]) for seq, s in sequence_to_split.items() if s == split
    )
    num_sequences = sum(1 for s in sequence_to_split.values() if s == split)
    total_num_segments = sum(len(entry["segs"]) for entry in layer.data.values())
    pct = (f" ({100 * num_segments / total_num_segments:.1f}%)"
           if total_num_segments and split != UNASSIGNED else "")
    label = layer.split_key(split)
    if not use_small_style:
        label = f"<strong>{label}</strong>"
    return _swatch_row(
        layer.colors[split],
        f"{label}: {num_segments} segments{pct}, {num_sequences} sequences",
        use_small_style=use_small_style,
    )


def _coverage_row_html(coverage: ClassCoverage, split: str) -> str:
    """Returns the HTML for the class-coverage line under a split's sidebar row.

    Reports only how many classes have no segment at all in the split, and how
    many of those any assignment of whole sequences could have covered. The
    shortfall against the minimum number of sequences is in the detail panel.

    Colouring follows the fixable count, not the raw one: a split missing only
    classes that are too rare to place everywhere has nothing to act on, and
    flagging it would be a false alarm that never clears.
    """
    num_absent = len(coverage.absent[split])
    num_fixable = coverage.num_absent_fixable(split)
    color = "var(--sidebar-text-dim)" if num_fixable else "var(--sidebar-text)"
    noun = "class" if num_absent == 1 else "classes"
    text = (f"{num_absent} {noun} absent ({num_fixable} fixable)"
            if num_absent else "no class absent")
    return (
        f'<div style="margin:0 0 6px 22px;font-size:0.8rem;color:{color};">'
        f"{text}"
        f"</div>"
    )


def _class_list_html(entries: list[str], limit: int = 12) -> str:
    """Returns a compact, scannable list of class descriptions.

    Entries are escaped: iRAP value names routinely contain ``<`` and ``>``
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


def _render_coverage_panel(
    coverage: ClassCoverage, class_names: dict[AttributeClass, str]
) -> None:
    """Renders the expandable class-coverage detail panel in the sidebar."""
    num_stratifiable = len(coverage.stratifiable)
    with st.sidebar.expander(
        f"Class coverage — {coverage.num_covered_everywhere}/{num_stratifiable} "
        f"classes in every split"
    ):
        # The value is in the session state under the key already (see `main`).
        st.number_input(
            "Min sequences per class per split",
            min_value=1,
            max_value=20,
            step=1,
            key=MIN_NUM_SEQUENCES_KEY,
            help=(
                "Support is counted in distinct sequences, not segments: "
                "consecutive segments of one road carry near-identical "
                "attributes, so many segments from a single section are one "
                "example, not many."
            ),
        )
        st.caption(
            f"{coverage.num_classes} (attribute, class) pairs occur in the data. "
            f"{num_stratifiable} of them appear in at least "
            f"{coverage.min_num_sequences_to_stratify} sequences, so some assignment "
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
            max_tvd = coverage.max_tvd[split]
            rows += (
                f'<tr><td style="color:{COLORS[split]};font-weight:600;">{split}</td>'
                f"<td>{len(coverage.absent[split])}</td>"
                f"<td>{coverage.num_absent_fixable(split)}</td>"
                f"<td>{len(coverage.short[split])}</td>"
                f"<td>{'–' if max_tvd is None else f'{max_tvd:.3f}'}</td></tr>"
            )
        st.markdown(
            f'<table style="font-size:0.78rem;width:100%;">{header}{rows}</table>',
            unsafe_allow_html=True,
        )

        for split in REAL_SPLITS:
            entries = [
                f"{num_sequences} seq / {num_segments} seg — {get_class_name(cls, class_names)} "
                f"[{coverage.class_to_num_sequences[cls]} seq total]"
                for num_sequences, num_segments, cls in coverage.short[split]
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
                    f"every stratifiable class has ≥{coverage.min_num_sequences} sequences</div>",
                    unsafe_allow_html=True,
                )

        if coverage.irreducible:
            st.markdown(
                f'<div style="margin:12px 0 4px 0;font-weight:600;color:var(--sidebar-text);">'
                f"{len(coverage.irreducible)} classes cannot be stratified</div>"
                f'<div style="font-size:0.75rem;color:var(--sidebar-text-dim);margin-bottom:4px;">'
                f"fewer than {coverage.min_num_sequences_to_stratify} sequences exist "
                f"dataset-wide; no assignment covers these in every split</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _class_list_html(
                    [
                        f"{num_sequences} seq — {get_class_name(cls, class_names)}"
                        for num_sequences, cls in coverage.irreducible
                    ]
                ),
                unsafe_allow_html=True,
            )


# App ##############################################################################################


def _parse_args() -> argparse.Namespace:
    # Streamlit passes everything after "--" to the script.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="iRAP-Vietnam dataset root.")
    return parser.parse_args()


def main() -> None:
    st.set_page_config(page_title="Split Editor", layout="wide")
    data_dir: Path = _parse_args().data_dir
    splits_path = layout.splits_path(data_dir)

    fingerprint = compute_metadata_fingerprint(data_dir)
    metadata = load_metadata(data_dir, fingerprint)
    labeled = Layer(
        name="labeled",
        data=metadata.sequence_data,
        colors=COLORS,
        marker_size=8,
        split_key_prefix="",
    )
    unlabeled = Layer(
        name="unlabeled",
        data=metadata.unlabeled_sequence_data,
        colors=UNLABELED_COLORS,
        marker_size=6,
        split_key_prefix=UNLABELED_SPLIT_PREFIX,
    )
    layers = (labeled, unlabeled)

    # Session state init. Keyed on the metadata fingerprint rather than mere
    # presence, so that re-running build_metadata.py under a live session
    # rebuilds the assignments instead of leaving them keyed by sequences that
    # no longer exist.
    if st.session_state.get("metadata_fingerprint") != fingerprint:
        st.session_state.metadata_fingerprint = fingerprint
        splits = _load_json(splits_path) if splits_path.exists() else {}
        st.session_state.assignments, warnings = make_initial_assignments(splits, layers)
        if metadata.num_stale_unlocated_seg_ids:
            num_seg_ids_in_file = (len(metadata.unlocated_seg_ids)
                                   + metadata.num_stale_unlocated_seg_ids)
            warnings.append(
                f"{layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME} is stale: "
                f"{metadata.num_stale_unlocated_seg_ids} of its {num_seg_ids_in_file} seg_ids "
                f"now belong to a real sequence. They are excluded from {UNLOCATED_SPLIT} so no "
                f"seg_id lands in two splits; re-run build_metadata.py to refresh the file."
            )
        st.session_state.load_warnings = warnings
        st.session_state.history = []  # earlier assignments, for Undo
    st.session_state.setdefault("map_key_version", 0)
    # Set before the coverage panel creates the widget with this key, as the
    # rows above the panel need the value too.
    st.session_state.setdefault(MIN_NUM_SEQUENCES_KEY, DEFAULT_MIN_NUM_SEQUENCES)

    assignments: Assignments = st.session_state.assignments

    # Sidebar
    st.sidebar.title("Split Editor")

    for warning in st.session_state.load_warnings:
        st.sidebar.warning(warning)

    active_split = st.sidebar.radio("Active split (draw a box to assign)", SPLITS)

    st.sidebar.markdown(PAGE_CSS, unsafe_allow_html=True)

    coverage = compute_class_coverage(
        labeled.data, assignments[labeled.name], st.session_state[MIN_NUM_SEQUENCES_KEY]
    )

    for split in SPLITS:
        st.sidebar.markdown(
            _sidebar_row(labeled, assignments[labeled.name], split, use_small_style=False),
            unsafe_allow_html=True,
        )
        if split in REAL_SPLITS and coverage.num_classes:
            st.sidebar.markdown(_coverage_row_html(coverage, split), unsafe_allow_html=True)
        if unlabeled.data:
            st.sidebar.markdown(
                _sidebar_row(unlabeled, assignments[unlabeled.name], split, use_small_style=True),
                unsafe_allow_html=True,
            )

    if metadata.unlocated_seg_ids:
        st.sidebar.markdown(
            _swatch_row(
                UNLABELED_COLORS[UNASSIGNED],
                f"{UNLOCATED_SPLIT}: {len(metadata.unlocated_seg_ids)} segments "
                f"(auto-assigned on save)",
                use_small_style=True,
            ),
            unsafe_allow_html=True,
        )

    if coverage.num_classes:
        _render_coverage_panel(coverage, metadata.class_names)

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    undo_clicked = col1.button("Undo", width="stretch", disabled=not st.session_state.history)
    reset_clicked = col2.button("Reset", width="stretch")

    st.sidebar.markdown("")
    save_clicked = st.sidebar.button(
        f"Save {layout.SPLITS_FILENAME}", width="stretch", type="primary"
    )

    # Handle sidebar actions before rendering the map
    if undo_clicked and st.session_state.history:
        st.session_state.assignments = st.session_state.history.pop()
        st.rerun()

    if reset_clicked:
        st.session_state.history.append(assignments)
        st.session_state.assignments = {
            layer.name: dict.fromkeys(layer.data, UNASSIGNED) for layer in layers
        }
        st.rerun()

    if save_clicked:
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(make_splits(layers, assignments, metadata.unlocated_seg_ids), f,
                      indent=2, ensure_ascii=False)
        # The file now matches the in-session state, so any load warning is stale.
        st.session_state.load_warnings = []
        st.sidebar.success(f"Saved {splits_path}")

    # Map
    event = st.plotly_chart(
        _build_figure(labeled, unlabeled, assignments),
        key=f"map_{st.session_state.map_key_version}",
        on_select="rerun",
        width="stretch",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "showTips": False,
            "responsive": True,
        },
    )

    # Handle map selection
    if event and event.selection and event.selection.points:
        new_assignments = assign_selected_sequences(
            assignments, event.selection.points, active_split
        )
        if new_assignments is not None:
            st.session_state.history.append(assignments)
            st.session_state.assignments = new_assignments
        # Streamlit keeps a chart's selection for as long as the chart's identity
        # (its key and figure) is unchanged. After a selection that changed
        # nothing, it would therefore stay drawn and be replayed on the next
        # rerun – into another split if that rerun came from the split radio.
        # A new key discards it.
        st.session_state.map_key_version += 1
        st.rerun()


if __name__ == "__main__":
    main()
