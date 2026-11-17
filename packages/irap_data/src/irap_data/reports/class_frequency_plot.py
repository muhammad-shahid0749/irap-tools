"""A bar chart of the class frequencies of all attributes of a segment set.

Needs matplotlib (the `report` extra). Figures are built without `pyplot`, so this module does not
change the global matplotlib backend and its figures need no closing.
"""

import typing as T
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .statistics import AttributeDistribution

#: Alternating shades of the attribute groups, so that group boundaries are visible without
#: relying on colour vision.
_GROUP_COLORS = ("#4878a8", "#a0c4e4")
_ABSENT_COLOR = "#c44e52"


def _get_bar_positions(
    distributions: T.Sequence[AttributeDistribution],
) -> tuple[np.ndarray, np.ndarray]:
    """The y positions of the class bars, one unit apart, with a row for the name of each
    attribute above its bars.

    Returns:
        `(positions, header_positions)`: the position of each class bar and of the name of each
        attribute.
    """
    sizes = np.array([d.num_classes for d in distributions])
    header_positions = np.concatenate([[0], np.cumsum(sizes[:-1] + 1)])
    positions = np.concatenate([start + 1 + np.arange(size)
                                for start, size in zip(header_positions, sizes)])
    return positions, header_positions


def plot_class_frequencies(
    distributions: T.Sequence[AttributeDistribution],
    *,
    title: str,
    normalize: bool = False,
    log: bool = True,
) -> Figure:
    """Plots the number of segments of each class as horizontal bars grouped by attribute.

    Attributes without a labeled segment are left out. A class without a segment is drawn as a
    short red bar, as zero has no position on a logarithmic axis and no width on a linear one.

    Args:
        distributions: The attributes to plot, in plot order (see `sort_distributions`).
        title: The figure title.
        normalize: Whether to plot the share of the labeled segments of the attribute instead
            of the number of segments, so that attributes are comparable.
        log: Whether the value axis is logarithmic.

    Raises:
        ValueError: If no attribute has a labeled segment.
    """
    distributions = [d for d in distributions if d.is_labeled]
    if not distributions:
        raise ValueError("No labeled attributes to plot.")

    positions, header_positions = _get_bar_positions(distributions)
    values = np.concatenate([d.labeled_shares if normalize else d.num_segments
                             for d in distributions]).astype(float)
    labels = [value for d in distributions for value in d.values]
    colors = [_ABSENT_COLOR if n == 0 else _GROUP_COLORS[group_index % len(_GROUP_COLORS)]
              for group_index, d in enumerate(distributions) for n in d.num_segments]

    positive = values[values > 0]
    absent_bar_length = positive.min() / 4 if log else values.max() / 200
    drawn = np.where(values > 0, values, absent_bar_length)

    num_rows = len(values) + len(distributions)
    height = max(4.0, 0.16 * num_rows + 1.2)
    # The constrained layout sizes the left margin to the longest class or attribute name.
    fig = Figure(figsize=(12, height), layout="constrained")
    ax = fig.subplots()
    ax.barh(positions, drawn, height=0.8, color=colors)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(left=absent_bar_length / 3)

    # The attribute names are tick labels of their own rows, so that they cannot overlap the
    # class names, whatever their length.
    ticks = np.concatenate([header_positions, positions])
    ax.set_yticks(ticks, [d.attribute for d in distributions] + labels, fontsize=6)
    for label in ax.get_yticklabels()[:len(distributions)]:
        label.set(fontsize=7, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(num_rows - 0.5, -0.5)  # The first attribute is at the top, without margins.
    ax.set_xlabel("share of the labeled segments of the attribute" if normalize
                  else "number of segments")
    ax.set_title(title)
    ax.grid(axis="x", which="both", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for position in header_positions[1:]:
        ax.axhline(position - 0.5, color="0.85", linewidth=0.6)

    if (values == 0).any():
        # A proxy patch, as the bars do not give the legend an artist of the absent colour. It
        # is below the axes, where it covers no bar.
        fig.legend(handles=[Patch(color=_ABSENT_COLOR, label="no segment")],
                   loc="outside lower right", fontsize=7)
    return fig


def write_class_frequency_plot(
    path: str | Path,
    distributions: T.Sequence[AttributeDistribution],
    *,
    title: str,
    normalize: bool = False,
    log: bool = True,
    dpi: int = 200,
) -> None:
    """Writes `plot_class_frequencies` to `path`, in the format of its extension (e.g. .pdf or
    .png)."""
    plot_class_frequencies(distributions, title=title, normalize=normalize, log=log).savefig(
        path, dpi=dpi)
