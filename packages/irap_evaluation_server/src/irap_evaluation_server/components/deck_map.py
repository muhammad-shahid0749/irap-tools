"""The map of the Analysis page: segments as points on a basemap, coloured by their outcome
(`prediction_analysis.AttributeOutcomes`), drawn by deck.gl in `deck_map.js`.

The libraries are served from `routes.VENDOR_PATH` (see `vendor_libraries`). The basemap tiles come
from Carto, so the map needs internet access.
"""

import html
import typing as T

import numpy as np
from nicegui import ui

from ..vendor_libraries import DECK_SCRIPT, MAPLIBRE_MODULE, MAPLIBRE_STYLESHEET
from .routes import get_library_path

BASEMAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

#: Outcome name (`prediction_analysis.MODEL_OUTCOME_NAMES` and `COMPARISON_OUTCOME_NAMES`) ->
#: (R, G, B), shared by the points, the legend and the confusion matrix.
OUTCOME_COLORS = {
    "correct": (42, 120, 214), "wrong": (235, 104, 52), "invalid": (142, 68, 173),
    "both correct": (42, 120, 214), "only A correct": (46, 160, 67),
    "only B correct": (235, 104, 52), "neither correct": (179, 38, 30),
    "unlabeled": (154, 163, 173),
}
#: Outcomes drawn smaller and paler, so that they do not hide the others.
_BACKGROUND_OUTCOMES = ("unlabeled",)


def _get_outcome_style(name: str) -> tuple[list[int], float]:
    """Returns ([R, G, B, alpha], radius in pixels) of the points of an outcome."""
    is_background = name in _BACKGROUND_OUTCOMES
    return [*OUTCOME_COLORS[name], 110 if is_background else 220], 2.5 if is_background else 4


class DeckMap(ui.element, component="deck_map.js"):
    """The map with its points (`deck_map.js`)."""

    def __init__(self, segment_ids: T.Sequence[str], coordinates: np.ndarray,
                 on_pick: T.Callable[[int], None]):
        """
        Args:
            segment_ids: (P,) of the points, for their tooltips.
            coordinates: (P, 2) latitude and longitude of each point in degrees.
            on_pick: Gets the index of a clicked point.
        """
        super().__init__()
        coordinates = np.asarray(coordinates).round(6)
        self._props.update({
            "latitudes": coordinates[:, 0].tolist(), "longitudes": coordinates[:, 1].tolist(),
            "segment-ids": list(segment_ids),
            "maplibre-url": get_library_path(MAPLIBRE_MODULE),
            "maplibre-stylesheet-url": get_library_path(MAPLIBRE_STYLESHEET),
            "deck-url": get_library_path(DECK_SCRIPT),
            "basemap-style-url": BASEMAP_STYLE_URL})
        self.on("pick", lambda e: on_pick(int(e.args)))

    def set_outcomes(self, point_outcomes: np.ndarray, outcome_names: T.Sequence[str]) -> None:
        """
        Args:
            point_outcomes: (P,) index into `outcome_names` of each point.
        """
        if len(outcome_names) > 10:
            raise ValueError("At most 10 outcomes can be sent as digits.")
        digits = (np.asarray(point_outcomes, dtype=np.uint8) + ord("0")).tobytes().decode("ascii")
        self.run_method("set_outcomes", digits, list(outcome_names),
                        [_get_outcome_style(n) for n in outcome_names])

    def set_highlighted(self, points: np.ndarray | None) -> None:
        """
        Args:
            points: The indices of the points to highlight, e.g. those of a matrix cell, or None.
                Unless it is None, the other points are faded.
        """
        self.run_method("set_highlighted", None if points is None else points.tolist())

    def set_selected(self, point: int | None) -> None:
        """
        Args:
            point: The index of the selected point, or None.
        """
        self.run_method("set_selected", point)


def make_map_legend_html(outcome_names: T.Sequence[str], counts: T.Sequence[int],
                         note: str = "") -> str:
    """Makes a legend of the colour and the number of points of each outcome that has points, and
    a note below them."""
    rows = "".join(
        f'<div><span class="swatch" style="background: rgb{OUTCOME_COLORS[name]}"></span>'
        f'{html.escape(name)} <span class="muted">{count}</span></div>'
        for name, count in zip(outcome_names, counts, strict=True) if count > 0)
    note_html = f'<div class="muted">{html.escape(note)}</div>' if note else ""
    return f'<div class="map-legend">{rows}{note_html}</div>'
