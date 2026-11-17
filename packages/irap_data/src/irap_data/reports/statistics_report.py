"""Reports of the statistics of an iRAP release: HTML pages, CSV and JSON.

The statistics come from `irap_data.reports.statistics`. The plots need matplotlib (the `report`
extra), which is imported only when a plot is written.
"""

import csv
import dataclasses as dc
import json
import typing as T
from pathlib import Path

import numpy as np

from . import document as rd
from . import segment_map
from ..metadata import IRAPMetadata, get_dataset_preset, is_unlabeled_split
from .statistics import (
    SELECTION_DESCRIPTIONS,
    AttributeDistribution,
    DatasetStatistics,
    SelectionKind,
    SelectionStatistics,
    SplitStatistics,
    compute_majority_accuracy,
    compute_selection_statistics,
    compute_total_variation_distance,
    select_segments,
    sort_distributions,
)

PlotFormat = T.Literal["pdf", "png", "svg"]

#: The columns of `to_class_frequency_rows`.
CLASS_FREQUENCY_COLUMNS = ("dataset", "selection", "split", "attribute", "class_index", "value",
                           "irap_code", "num_segments", "num_sequences", "share", "share_labeled")


# Formatting helpers ###############################################################################

def _format_context_offsets(context_offsets: T.Sequence[int]) -> str:
    """Formats consecutive ascending offsets as a range, e.g. '-4..4', and others as a list."""
    first, last = context_offsets[0], context_offsets[-1]
    if len(context_offsets) > 2 and list(context_offsets) == list(range(first, last + 1)):
        return f"{first}..{last}"
    return ",".join(map(str, context_offsets))


def _format_selection_name(selection: SelectionKind) -> str:
    """The name of a selection in a report, e.g. 'with context' for 'with_context'."""
    return selection.replace("_", " ")


def _describe_selection(selection: SelectionKind,
                        context_offsets: T.Sequence[int] | None) -> str:
    """'labeled segments', or 'labeled segments with context [-4..0]' for a selection with a
    context."""
    if context_offsets is None:
        return f"{selection} segments"
    return f"labeled segments with context [{_format_context_offsets(context_offsets)}]"


def _get_report_stem(dataset: str, selection: SelectionKind,
                     context_offsets: T.Sequence[int] | None) -> str:
    """The file name stem of the report of a selection: 'vietnam_labeled_segments',
    'vietnam_labeled_segments_with_context' for the with-context selection, and
    'vietnam_labeled_segments_with_context_<context_offsets>' for the model selection."""
    if selection == "labeled":
        return f"{dataset}_labeled_segments"
    stem = f"{dataset}_labeled_segments_with_context"
    if selection == "with_context":
        return stem
    return f"{stem}_{_format_context_offsets(context_offsets)}"


def _format_count(n: int) -> str:
    return f"{n:,}"


def _format_share(x: float) -> str:
    """A share with one decimal, with a bound for a share that would round to 0% or 100%
    without being either, e.g. 2 missing labels of 10,000."""
    if np.isnan(x):
        return ""
    if 0 < x < 0.0005:
        return "<0.1%"
    if 0.9995 <= x < 1:
        return ">99.9%"
    return f"{x:.1%}"


def _format_ratio(x: float) -> str:
    if np.isnan(x):
        return ""
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def _format_class_count(n: int, share: float | None = None) -> str:
    """A class count, with its share if given and the count is not zero."""
    if n == 0 or share is None:
        return _format_count(n)
    return f"{_format_count(n)} ({_format_share(share)})"


def _to_json_float(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


# Report ###########################################################################################

def compose_report(
    statistics: DatasetStatistics,
    selection: SelectionKind,
    *,
    report_date: str,
    rare_fraction: float,
    split_to_figure_path: T.Mapping[str, str] | None = None,
    map_path: str | None = None,
) -> list[rd.Block]:
    """Composes the report of one selection of segments, to be rendered with `document.to_html`.

    The class balance is measured on the base split, the first split of `statistics` with
    segments in the selection, and the other splits are compared to it.

    Args:
        report_date: The date of the report, YYYY-MM-DD.
        rare_fraction: The share of the labeled segments of an attribute in the base split below
            which a class is listed as rare.
        split_to_figure_path: Split -> the path of its class-frequency plot, relative to the
            report.
        map_path: The path of the segment map (`segment_map.to_segment_map_html`), relative to
            the report, or None for no map.
    """
    split_distributions = statistics.get_split_distributions(selection)
    context_offsets = statistics.get_context_offsets(selection)
    description = _describe_selection(selection, context_offsets)
    blocks = [*_compose_header(statistics, selection, description, report_date),
              *_compose_overview(statistics, selection, description),
              *([] if map_path is None else _compose_map(context_offsets, map_path))]
    if not split_distributions:
        return [*blocks, rd.Paragraph("No split has segments in this selection.")]
    num_attributes = len(next(iter(split_distributions.values())))
    attribute_indices = [i for i in range(num_attributes)
                         if any(ds[i].is_labeled for ds in split_distributions.values())]
    return [
        *blocks,
        *_compose_label_coverage(split_distributions, attribute_indices),
        *_compose_class_balance(split_distributions, attribute_indices),
        *_compose_rare_classes(split_distributions, attribute_indices, rare_fraction),
        *_compose_missing_classes(split_distributions, attribute_indices),
        *_compose_class_frequencies(split_distributions, attribute_indices,
                                    split_to_figure_path or {}),
    ]


def _compose_header(statistics: DatasetStatistics, selection: SelectionKind, description: str,
                    report_date: str) -> list[rd.Block]:
    preset = statistics.preset
    items = [
        f"Date: {report_date}",
        f"Selection: {_format_selection_name(selection)}. {SELECTION_DESCRIPTIONS[selection]}",
        "A segment with a missing attribute code is "
        + ("kept without a label for the attribute." if preset.allow_missing_attributes
           else "left out."),
    ]
    if preset.use_ncontext_filter and selection != "labeled":
        items.append("The segments are restricted to the precomputed N-context subsets"
                     " (`seg_to_res`).")
    return [rd.Heading(f"Dataset statistics of {statistics.dataset}: {description}", 1),
            rd.BulletList(tuple(items))]


def _compose_overview(statistics: DatasetStatistics, selection: SelectionKind,
                      description: str) -> list[rd.Block]:
    selections = statistics.selections

    def format_row(name: str, splits: T.Sequence[SplitStatistics]) -> list[str]:
        """The cells of a split, or the totals of several splits."""
        def format_total(counts: T.Sequence[int]) -> str:
            return _format_count(sum(counts)) if counts else ""

        selected = [s.selections[selection] for s in splits if s.selections]
        lengths = np.concatenate([s.sequence_lengths for s in selected] or [[]]).astype(np.int64)
        sequence_cells = [
            format_total([s.num_sequences for s in selected]),
            f"{lengths.min()} / {np.median(lengths):g} / {lengths.max()}" if lengths.size else "",
            f"{sum(s.road_length_m for s in selected) / 1000:,.1f}"] if selected else [""] * 3
        return [name, format_total([s.num_listed for s in splits]),
                format_total([s.num_with_image for s in splits]),
                *(format_total([s.num_selected[k] for s in splits if k in s.num_selected])
                  for k in selections),
                *sequence_cells]

    labeled = [s for s in statistics.splits if not is_unlabeled_split(s.split)]
    unlabeled = [s for s in statistics.splits if is_unlabeled_split(s.split)]
    # Subtotals only for a release with both labeled and unlabeled splits.
    subtotals = ([format_row("labeled splits", labeled),
                  format_row("unlabeled splits", unlabeled)] if labeled and unlabeled else [])
    table = rd.make_table(
        ["split", "listed", "with image", *map(_format_selection_name, selections),
         "sequences", "segments per sequence (min / median / max)", "road length (km)"],
        [format_row(s.split, [s]) for s in statistics.splits],
        "l" + "r" * (5 + len(selections)),
        footer=[*subtotals, format_row("total", statistics.splits)])
    return [
        rd.Heading("Overview", 2),
        rd.Paragraph(
            "The number of segments of each split: listed in `splits.json`, with an image, and"
            " in each selection. A selection with context leaves out the segments whose context"
            " window is incomplete: it reaches beyond the road sequence of the segment, as for"
            " the first segments of a road with negative offsets, or it has a segment without an"
            " image. For an unlabeled split, it counts the segments that a dataset of the split"
            " yields, without the unlabeled segments that are in no road sequence. The last"
            " three columns describe the"
            f" {description}: their"
            " road sequences in `road_id_to_segment_id_sequence.json`, where a segment in none"
            " is a sequence of its own, and the summed length of those with road data."
            " The totals add up the splits, so a road sequence with segments in two splits is"
            " counted once in each."),
        table]


def _compose_map(context_offsets: T.Sequence[int] | None, map_path: str) -> list[rd.Block]:
    text = ("The segments of each split, labeled and unlabeled, colored by split. The segments of"
            " an unlabeled split are placed between the labeled segments of their road"
            " sequence.")
    if context_offsets is not None:
        text += (f" The segments that are not in the selection, as their context window"
                 f" [{_format_context_offsets(context_offsets)}] is incomplete, are faded.")
    text += " The map loads its basemap and the Leaflet library from the web."
    return [rd.Heading("Map", 2), rd.Paragraph(text), rd.Figure(map_path, "Segment map")]


def _compose_label_coverage(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    def format_cell(d: AttributeDistribution) -> str:
        if not d.num_missing:
            return _format_count(d.num_labeled)
        return (f"{_format_count(d.num_labeled)}"
                f" ({_format_share(d.num_missing / d.num_total)} missing)")

    distributions_seq = list(split_distributions.values())
    table = rd.make_table(
        ["attribute", *split_distributions],
        [[distributions_seq[0][i].attribute, *(format_cell(ds[i]) for ds in distributions_seq)]
         for i in attribute_indices],
        "l" + "r" * len(split_distributions))
    blocks = [rd.Heading("Label coverage", 2),
              rd.Paragraph("The number of labeled segments of each attribute, and the share of"
                           " the segments without a label."),
              table]
    if unlabeled := [d.attribute for i, d in enumerate(distributions_seq[0])
                     if i not in attribute_indices]:
        blocks.append(rd.Paragraph(f"Attributes without a label in any split, left out below"
                                   f" ({len(unlabeled)}): {', '.join(unlabeled)}."))
    return blocks


def _compose_class_balance(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    base_split, *other_splits = split_distributions
    base = split_distributions[base_split]

    def format_row(i: int) -> list[str]:
        d = base[i]
        comparisons = [(compute_total_variation_distance(d, split_distributions[s][i]),
                        compute_majority_accuracy(d, split_distributions[s][i]))
                       for s in other_splits]
        return [d.attribute, str(d.num_classes), str(d.num_absent_classes),
                _format_share(d.majority_share), _format_ratio(d.imbalance_ratio),
                "" if np.isnan(d.effective_num_classes) else f"{d.effective_num_classes:.1f}",
                *(_format_share(x) for comparison in comparisons for x in comparison)]

    table = rd.make_table(
        ["attribute", "classes", "absent", "majority share", "max:min", "effective classes",
         *(f"{name} {s}" for s in other_splits for name in ("TVD", "majority accuracy"))],
        map(format_row, attribute_indices),
        "lrrrrr" + "rr" * len(other_splits))
    legend = (
        f"Measured on {base_split}. *Absent* is the number of classes without a segment."
        " *Majority share* is the share of the most frequent class, the accuracy of predicting"
        " it. *Max:min* is the ratio of the most to the least frequent class that has a segment."
        " *Effective classes* is the exponential of the class entropy: K for K equally frequent"
        " classes, and 1 for a single class.")
    if other_splits:
        legend += (
            f" *TVD* is the total variation distance between the class distributions of a split"
            f" and of {base_split}: the share of labels that would have to change class for them"
            f" to match. *Majority accuracy* is the accuracy on a split of predicting the most"
            f" frequent class of {base_split}.")
    return [rd.Heading("Class balance", 2), rd.Paragraph(legend), table]


def _compose_rare_classes(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
    rare_fraction: float,
) -> list[rd.Block]:
    base_split = next(iter(split_distributions))
    base = split_distributions[base_split]
    distributions_seq = list(split_distributions.values())
    rare = sorted(((i, c) for i in attribute_indices for c in range(base[i].num_classes)
                   if 0 < base[i].labeled_shares[c] < rare_fraction),
                  key=lambda item: (base[item[0]].num_segments[item[1]], base[item[0]].attribute))
    table = rd.make_table(
        ["attribute", "class", "iRAP code", *(f"{s} segments" for s in split_distributions),
         f"{base_split} share", f"{base_split} sequences"],
        [[base[i].attribute, base[i].values[c], str(base[i].irap_codes[c]),
          *(_format_class_count(ds[i].num_segments[c]) for ds in distributions_seq),
          _format_share(base[i].labeled_shares[c]), _format_count(base[i].num_sequences[c])]
         for i, c in rare],
        "llr" + "r" * len(split_distributions) + "rr")
    return [
        rd.Heading("Rare classes", 2),
        rd.Paragraph(
            f"The classes with less than {rare_fraction:.1%} of the labeled segments of their"
            f" attribute in {base_split}, rarest first, with their number of segments in each"
            f" split. Classes without a segment are in the next section."),
        table if rare else rd.Paragraph("None.")]


def _compose_missing_classes(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    base_split = next(iter(split_distributions))
    distributions_seq = list(split_distributions.values())
    rows = [[d.attribute, d.values[c], str(d.irap_codes[c]),
             *(_format_class_count(ds[i].num_segments[c]) for ds in distributions_seq)]
            for i in attribute_indices for d in [distributions_seq[0][i]]
            for c in range(d.num_classes)
            if any(ds[i].num_segments[c] == 0 for ds in distributions_seq)]
    table = rd.make_table(["attribute", "class", "iRAP code", *split_distributions], rows,
                          "llr" + "r" * len(split_distributions))
    return [
        rd.Heading("Missing classes", 2),
        rd.Paragraph(
            f"The classes without a segment in at least one split. A class that is missing from"
            f" {base_split} cannot be learned from it, and the recall of a class that is missing"
            f" from another split cannot be measured on it."),
        table if rows else rd.Paragraph("None.")]


def _compose_class_frequencies(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
    split_to_figure_path: T.Mapping[str, str],
) -> list[rd.Block]:
    blocks = [rd.Heading("Class frequencies", 2),
              rd.Paragraph("The number of segments of each class, with its share of the labeled"
                           " segments of the attribute, and the number of road sequences that"
                           " contain the class.")]
    blocks += [rd.Figure(path, f"Class frequencies of {split}")
               for split, path in split_to_figure_path.items()]
    distributions_seq = list(split_distributions.values())
    for i in attribute_indices:
        first = distributions_seq[0][i]
        table = rd.make_table(
            ["class", "iRAP code",
             *(name for s in split_distributions for name in (s, f"{s} sequences"))],
            [[first.values[c], str(first.irap_codes[c]),
              *(cell for ds in distributions_seq
                for cell in (_format_class_count(ds[i].num_segments[c], ds[i].labeled_shares[c]),
                             _format_count(ds[i].num_sequences[c])))]
             for c in range(first.num_classes)],
            "lr" + "rr" * len(split_distributions))
        blocks += [rd.Heading(first.attribute, 3), table]
    return blocks


# JSON and CSV #####################################################################################

def to_json_dict(statistics: DatasetStatistics, *, report_date: str) -> dict[str, T.Any]:
    """A JSON-compatible document of all statistics, with NaN as None."""
    def distribution_to_json(d: AttributeDistribution) -> dict[str, T.Any]:
        return {
            "attribute": d.attribute, "num_labeled": d.num_labeled, "num_missing": d.num_missing,
            "num_absent_classes": d.num_absent_classes,
            "majority_share": _to_json_float(d.majority_share),
            "imbalance_ratio": _to_json_float(d.imbalance_ratio),
            "effective_num_classes": _to_json_float(d.effective_num_classes),
            "classes": [{"index": c, "value": d.values[c], "irap_code": d.irap_codes[c],
                         "num_segments": int(d.num_segments[c]),
                         "num_sequences": int(d.num_sequences[c]),
                         "share": _to_json_float(d.shares[c]),
                         "share_labeled": _to_json_float(d.labeled_shares[c])}
                        for c in range(d.num_classes)],
        }

    def selection_to_json(s: SelectionStatistics) -> dict[str, T.Any]:
        return {"num_segments": s.num_segments, "num_unlabeled": s.num_unlabeled,
                "num_without_sequence": s.num_without_sequence,
                "num_sequences": s.num_sequences, "road_length_m": s.road_length_m,
                "attributes": [distribution_to_json(d) for d in s.distributions]}

    def get_context_offsets(selection: SelectionKind) -> list[int] | None:
        offsets = statistics.get_context_offsets(selection)
        return None if offsets is None else list(offsets)

    return {
        "dataset": statistics.dataset,
        "metadata_dir": str(statistics.metadata_dir),
        "date": report_date,
        "preset": dc.asdict(statistics.preset),
        "selections": {k: {"description": SELECTION_DESCRIPTIONS[k],
                           "context_offsets": get_context_offsets(k)}
                       for k in statistics.selections},
        "splits": [{"split": s.split, "num_listed": s.num_listed,
                    "num_with_image": s.num_with_image, "num_selected": dict(s.num_selected),
                    "selections": {k: selection_to_json(v) for k, v in s.selections.items()}}
                   for s in statistics.splits],
    }


def to_class_frequency_rows(statistics: DatasetStatistics,
                            selection: SelectionKind) -> list[dict[str, T.Any]]:
    """One row per (split with segments in the selection, attribute, class), with the columns
    `CLASS_FREQUENCY_COLUMNS`. `share` is the share of all segments of the split, and
    `share_labeled` the share of its labeled segments of the attribute, None for an attribute
    without a label."""
    return [{"dataset": statistics.dataset, "selection": selection, "split": split,
             "attribute": d.attribute, "class_index": c, "value": d.values[c],
             "irap_code": d.irap_codes[c], "num_segments": int(d.num_segments[c]),
             "num_sequences": int(d.num_sequences[c]), "share": _to_json_float(d.shares[c]),
             "share_labeled": _to_json_float(d.labeled_shares[c])}
            for split, distributions in statistics.get_split_distributions(selection).items()
            for d in distributions for c in range(d.num_classes)]


# Files ############################################################################################

def _import_class_frequency_plot():
    try:
        from . import class_frequency_plot
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Plots need {e.name!r}. Install the extra: pip install 'irap-data[report]'.") from e
    return class_frequency_plot


def _format_plot_title(dataset: str, split: str, selection_description: str,
                       num_segments: int) -> str:
    return (f"{dataset} / {split} – {selection_description} ({num_segments:,}), class"
            f" frequencies")


def write_selection_report(
    statistics: DatasetStatistics,
    selection: SelectionKind,
    out_dir: Path,
    *,
    report_date: str,
    metadata: IRAPMetadata | None = None,
    plot_format: PlotFormat | None = "svg",
    rare_fraction: float = 0.01,
    dpi: int = 200,
) -> list[Path]:
    """Writes the HTML report of one selection of segments to `out_dir`, with a class-frequency
    plot of each labeled split, a CSV table of the class frequencies (`to_class_frequency_rows`)
    and a map of the segments of all splits of `statistics` (`segment_map`).

    Args:
        report_date: The date of the report, YYYY-MM-DD.
        metadata: The metadata that `statistics` are computed from, for the map, or None for no
            map.
        plot_format: The file format of the plots, or None for no plots. The report shows each
            format as `document.Figure` describes.
        rare_fraction: See `compose_report`.
        dpi: The resolution of raster plots.

    Returns:
        The written paths.
    """
    class_frequency_plot = None if plot_format is None else _import_class_frequency_plot()
    context_offsets = statistics.get_context_offsets(selection)
    stem = _get_report_stem(statistics.dataset, selection, context_offsets)
    description = _describe_selection(selection, context_offsets)
    written = []
    split_to_figure_path = {}
    for split in statistics.splits if class_frequency_plot else ():
        selected = split.selections.get(selection)
        if selected is None or not any(d.is_labeled for d in selected.distributions):
            continue
        path = out_dir / f"{stem}_{split.split}_class_frequencies.{plot_format}"
        class_frequency_plot.write_class_frequency_plot(
            path, selected.distributions, dpi=dpi,
            title=_format_plot_title(statistics.dataset, split.split, description,
                                     selected.num_segments))
        split_to_figure_path[split.split] = path.name
        written.append(path)

    map_path = None
    if metadata is not None:
        path = out_dir / f"{stem}_map.html"
        split_maps = segment_map.compute_split_segment_maps(metadata, statistics, selection)
        path.write_text(segment_map.to_segment_map_html(
            split_maps, title=f"Segment map of {statistics.dataset}: {description}",
            description=SELECTION_DESCRIPTIONS[selection]), encoding="utf-8")
        map_path = path.name
        written.append(path)

    blocks = compose_report(statistics, selection, report_date=report_date,
                            rare_fraction=rare_fraction, split_to_figure_path=split_to_figure_path,
                            map_path=map_path)
    path = out_dir / f"{stem}.html"
    path.write_text(rd.to_html(blocks), encoding="utf-8")
    written.append(path)

    path = out_dir / f"{stem}_class_frequencies.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLASS_FREQUENCY_COLUMNS)
        writer.writeheader()
        writer.writerows(to_class_frequency_rows(statistics, selection))
    written.append(path)
    return written


def write_statistics_reports(
    statistics: DatasetStatistics,
    out_dir: Path,
    *,
    report_date: str,
    selections: T.Sequence[SelectionKind] | None = None,
    **selection_report_kwargs,
) -> list[Path]:
    """Writes the report of each selection with `write_selection_report`, and all statistics to
    `<dataset>_statistics.json` (`to_json_dict`).

    The JSON file is written last. `write_selection_report` imports matplotlib before it writes
    anything, so a missing matplotlib leaves no partial output.

    Args:
        report_date: The date of the reports, YYYY-MM-DD.
        selections: Default: all selections of `statistics`, without 'model' if it repeats
            'with_context' (see `DatasetStatistics.is_model_selection_same_as_with_context`).
        **selection_report_kwargs: Passed to `write_selection_report`.

    Returns:
        The written paths.
    """
    if selections is None:
        selections = tuple(s for s in statistics.selections if not (
            s == "model" and statistics.is_model_selection_same_as_with_context))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [path for selection in selections
               for path in write_selection_report(statistics, selection, out_dir,
                                                  report_date=report_date,
                                                  **selection_report_kwargs)]
    path = out_dir / f"{statistics.dataset}_statistics.json"
    path.write_text(json.dumps(to_json_dict(statistics, report_date=report_date), indent=2,
                               ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return [*written, path]


def write_split_class_frequency_plot(
    path: Path,
    metadata: IRAPMetadata,
    dataset_name: str,
    split: str,
    *,
    selection: SelectionKind = "labeled",
    context_offsets: T.Sequence[int] | None = None,
    attribute_order: str = "schema",
    class_order: str = "schema",
    normalize: bool = False,
    log: bool = True,
    dpi: int = 200,
) -> None:
    """Plots the class frequencies of all attributes of one split in one figure, in the format
    given by the extension of `path`.

    Args:
        dataset_name: A key of `DATASET_PRESETS`.
        selection: The segments to count (see `SELECTION_DESCRIPTIONS`).
        context_offsets: The context offsets of the 'model' selection, and only of it.
        attribute_order: One of `ATTRIBUTE_ORDERS`.
        class_order: One of `CLASS_ORDERS`.
        normalize: Plot the share of the labeled segments of the attribute instead of the number
            of segments.
        log: Use a logarithmic value axis.
        dpi: The resolution of a raster plot.

    Raises:
        ValueError: For an unlabeled split, and see `select_segments` and `sort_distributions`.
    """
    class_frequency_plot = _import_class_frequency_plot()
    if is_unlabeled_split(split):
        raise ValueError(f"The split {split!r} has no labels.")
    selected = compute_selection_statistics(metadata, select_segments(
        metadata, dataset_name, split, selection, context_offsets))
    if selection == "with_context":
        context_offsets = get_dataset_preset(dataset_name).reference_context_offsets
    distributions = sort_distributions(selected.distributions, attribute_order=attribute_order,
                                       class_order=class_order)
    class_frequency_plot.write_class_frequency_plot(
        path, distributions, normalize=normalize, log=log, dpi=dpi,
        title=_format_plot_title(dataset_name, split, _describe_selection(
            selection, context_offsets), selected.num_segments))
