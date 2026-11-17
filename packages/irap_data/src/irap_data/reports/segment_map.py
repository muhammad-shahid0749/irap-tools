"""Maps of the segments of the splits of an iRAP release, as interactive HTML pages.

A map shows where the segments of each split are, labeled and unlabeled, and which of them a
selection of segments leaves out (see `irap_data.reports.statistics.SELECTION_DESCRIPTIONS`),
e.g. the segments near the start of a road that have no complete context window. The page draws
the segments with Leaflet on the Esri gray canvas basemap, which it loads from the web when it is
opened. Without a connection, it shows nothing.
"""

import dataclasses as dc
import html
import json
import typing as T

import numpy as np

from .document import THEME_CSS, to_html_inline
from ..metadata import IRAPMetadata, compute_split_labels, is_unlabeled_split
from .statistics import DatasetStatistics, SelectionKind, SplitStatistics

#: Categorical colors of the split groups (a labeled split and its `unlabeled_` split share
#: one), for the light and the dark basemap. The first three are distinguishable with color
#: vision deficiencies when all pairs are plotted together, as on a map.
_LIGHT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                 "#e34948")
_DARK_COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
                "#e66767")

#: Leaflet 1.9.4, with the hashes that the browser verifies.
_LEAFLET_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet"
_LEAFLET_JS_SRI = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
_LEAFLET_CSS_SRI = "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="

#: Coordinates are rounded to 5 decimals, about 1 m, to keep the page small.
_COORDINATE_DECIMALS = 5


# Map data #########################################################################################

@dc.dataclass(frozen=True, eq=False)
class SplitSegmentMap:
    """The segments of one split on a map.

    Attributes:
        split: The split name.
        segment_ids: (N,) the segments of the split that have an image and labels (see
            `irap_data.metadata.compute_split_labels`) and a location (see
            `irap_data.metadata.IRAPMetadata.segment_coordinates`), in `splits.json` order.
        coordinates: (N, 2) float64 latitudes and longitudes in degrees.
        is_selected: (N,) bool, whether each segment is in the selection.
        num_unplaced: The number of segments with an image and labels but without a location.
    """

    split: str
    segment_ids: tuple[str, ...]
    coordinates: np.ndarray
    is_selected: np.ndarray
    num_unplaced: int

    @property
    def is_labeled(self) -> bool:
        return not is_unlabeled_split(self.split)

    @property
    def group(self) -> str:
        """The split without the `unlabeled_` prefix, e.g. 'train' for 'unlabeled_train'."""
        return self.split.removeprefix("unlabeled_")


def compute_split_segment_maps(
    metadata: IRAPMetadata,
    statistics: DatasetStatistics,
    selection: SelectionKind,
) -> tuple[SplitSegmentMap, ...]:
    """Places the segments of each split of `statistics` on a map and marks those in `selection`.

    The segments of an unlabeled split are selected as the loaders select them: all that have
    an image for the 'labeled' selection, and those with a complete context window for the
    others (see `SplitStatistics.selection_to_segment_ids`).

    Args:
        metadata: The metadata that `statistics` are computed from.

    Raises:
        ValueError: If `statistics` have no `selection`.
    """
    if selection not in statistics.selections:
        raise ValueError(f"The statistics have no {selection!r} selection.")
    allow_missing_attributes = statistics.preset.allow_missing_attributes
    coordinates = metadata.segment_coordinates

    def compute_split_map(split: SplitStatistics) -> SplitSegmentMap:
        candidates = compute_split_labels(metadata, split.split,
                                          allow_missing_attributes=allow_missing_attributes)
        # An unlabeled split has no 'labeled' selection. All its candidates are selected.
        selected = set(split.selection_to_segment_ids.get(selection, candidates))
        placed = tuple(sid for sid in candidates if sid in coordinates)
        return SplitSegmentMap(
            split=split.split, segment_ids=placed,
            coordinates=np.array([coordinates[sid] for sid in placed],
                                 dtype=np.float64).reshape(len(placed), 2),
            is_selected=np.array([sid in selected for sid in placed], dtype=bool),
            num_unplaced=len(candidates) - len(placed))

    return tuple(map(compute_split_map, statistics.splits))


# HTML page ########################################################################################

def _get_group_to_color_index(split_maps: T.Sequence[SplitSegmentMap]) -> dict[str, int]:
    groups = list(dict.fromkeys(m.group for m in split_maps))
    if len(groups) > len(_LIGHT_COLORS):
        raise ValueError(f"A map can show at most {len(_LIGHT_COLORS)} split groups, not"
                         f" {len(groups)}: {groups}.")
    return {group: i for i, group in enumerate(groups)}


def _to_map_data(split_maps: T.Sequence[SplitSegmentMap]) -> dict[str, T.Any]:
    group_to_color_index = _get_group_to_color_index(split_maps)
    coordinates = [np.round(m.coordinates, _COORDINATE_DECIMALS) for m in split_maps]
    return {
        "lightColors": _LIGHT_COLORS, "darkColors": _DARK_COLORS,
        "splits": [{"split": m.split, "isLabeled": m.is_labeled,
                    "colorIndex": group_to_color_index[m.group], "ids": m.segment_ids,
                    "lat": c[:, 0].tolist(), "lon": c[:, 1].tolist(),
                    "selected": m.is_selected.astype(int).tolist()}
                   for m, c in zip(split_maps, coordinates) if m.segment_ids],
    }


def _compose_notes(split_maps: T.Sequence[SplitSegmentMap]) -> list[str]:
    notes = []
    if unplaced := [f"{m.split} ({m.num_unplaced:,})" for m in split_maps if m.num_unplaced]:
        notes.append("Not on the map, as they have no location: " + ", ".join(unplaced) + ".")
    if empty := [m.split for m in split_maps if not m.segment_ids and not m.num_unplaced]:
        notes.append("Splits without segments: " + ", ".join(empty) + ".")
    return notes


def to_segment_map_html(split_maps: T.Sequence[SplitSegmentMap], *, title: str,
                        description: str) -> str:
    """An HTML page with an interactive map of the segments of `split_maps`.

    A segment of a labeled split is a filled dot, one of an unlabeled split a ring, in the color
    of its split group. A segment that is not in the selection is faded. Each split is a layer
    that can be hidden, and a click shows the segment ID.

    Args:
        title: The page heading.
        description: A sentence or two under the heading, e.g. what the selection is, with
            the inline Markdown of `irap_data.reports.document`.

    Raises:
        ValueError: If there are more split groups than colors.
    """
    data = json.dumps(_to_map_data(split_maps), separators=(",", ":")).replace("</", "<\\/")
    notes = "".join(f"<p>{html.escape(note)}</p>" for note in _compose_notes(split_maps))
    return (_HTML_TEMPLATE
            .replace("$THEME_CSS", THEME_CSS)
            .replace("$TITLE", html.escape(title))
            .replace("$DESCRIPTION", to_html_inline(description))
            .replace("$NOTES", notes)
            .replace("$LEAFLET_URL", _LEAFLET_URL)
            .replace("$LEAFLET_JS_SRI", _LEAFLET_JS_SRI)
            .replace("$LEAFLET_CSS_SRI", _LEAFLET_CSS_SRI)
            .replace("$DATA", data))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$TITLE</title>
<link rel="stylesheet" href="$LEAFLET_URL.css" integrity="$LEAFLET_CSS_SRI" crossorigin="">
<style>
$THEME_CSS
html, body { height: 100%; margin: 0; }
body { display: flex; flex-direction: column; color: var(--fg); background: var(--bg);
       font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 8px 16px; border-bottom: 1px solid var(--border); }
header h1 { margin: 0 0 4px; font-size: 1.2em; }
header p { margin: 2px 0; }
code { font-size: 0.9em; }
#map { flex: 1; min-height: 300px; }
.leaflet-container { font: inherit; }
.legend, .leaflet-control-layers { color: #1f2328; background: rgba(255, 255, 255, 0.92); }
.legend { padding: 6px 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0, 0, 0, 0.4); }
.legend div, .leaflet-control-layers label { display: flex; align-items: center; gap: 6px; }
.swatch { display: inline-block; flex: none; width: 10px; height: 10px; border-radius: 50%;
          box-sizing: border-box; }
</style>
</head>
<body>
<header>
<h1>$TITLE</h1>
<p>$DESCRIPTION</p>
$NOTES
</header>
<div id="map"></div>
<script src="$LEAFLET_URL.js" integrity="$LEAFLET_JS_SRI" crossorigin=""></script>
<script>
const DATA = $DATA;
const FADED_OPACITY = 0.3;
const CLICK_RADIUS_PX = 10;

const isDark = matchMedia("(prefers-color-scheme: dark)").matches;
const colors = isDark ? DATA.darkColors : DATA.lightColors;
const map = L.map("map", {preferCanvas: true});
// The basemap has tiles up to zoom 16 and is magnified beyond, where 20 m segments get apart.
const basemap = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_"
                + (isDark ? "Dark" : "Light") + "_Gray";
const tileOptions = {maxNativeZoom: 16, maxZoom: 20};
L.tileLayer(`${basemap}_Base/MapServer/tile/{z}/{y}/{x}`, {...tileOptions,
  attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors,"
               + " and the GIS user community",
}).addTo(map);
L.tileLayer(`${basemap}_Reference/MapServer/tile/{z}/{y}/{x}`, tileOptions).addTo(map);
const renderer = L.canvas({padding: 0.5});

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
}

function makeSwatch(color, isLabeled, opacity) {
  const style = isLabeled ? `background: ${color}` : `border: 2px solid ${color}`;
  return `<span class="swatch" style="${style}; opacity: ${opacity}"></span>`;
}

const overlays = {};
const splitLayers = [];
for (const s of DATA.splits) {
  const color = colors[s.colorIndex];
  const layer = L.layerGroup();
  // Faded segments first, so that the selected ones are drawn over them.
  for (const drawSelected of [0, 1]) {
    for (let i = 0; i < s.ids.length; i++) {
      if (s.selected[i] !== drawSelected) continue;
      const opacity = drawSelected ? 1 : FADED_OPACITY;
      L.circleMarker([s.lat[i], s.lon[i]], s.isLabeled
        ? {renderer, interactive: false, radius: 4, stroke: false, fillColor: color,
           fillOpacity: opacity}
        : {renderer, interactive: false, radius: 3.5, weight: 1.5, color, opacity,
           fill: false}).addTo(layer);
    }
  }
  layer.addTo(map);
  const numSelected = s.selected.reduce((a, b) => a + b, 0);
  const label = `${makeSwatch(color, s.isLabeled, 1)} ${escapeHtml(s.split)}: `
                + `${numSelected.toLocaleString("en")} of ${s.ids.length.toLocaleString("en")}`
                + " in the selection";
  overlays[label] = layer;
  splitLayers.push({split: s, layer});
}
L.control.layers(null, overlays, {collapsed: false}).addTo(map);

const legend = L.control({position: "bottomright"});
legend.onAdd = () => {
  const div = L.DomUtil.create("div", "legend");
  const gray = "#59636e";
  div.innerHTML = `<div>${makeSwatch(gray, true, 1)} segment of a labeled split</div>`
    + `<div>${makeSwatch(gray, false, 1)} segment of an unlabeled split</div>`
    + `<div>${makeSwatch(gray, true, FADED_OPACITY)} not in the selection</div>`;
  return div;
};
legend.addTo(map);

// The markers are not interactive, which keeps panning fast. A click finds the nearest one.
map.on("click", event => {
  let best = null;
  let bestDistance = CLICK_RADIUS_PX;
  for (const {split: s, layer} of splitLayers) {
    if (!map.hasLayer(layer)) continue;
    for (let i = 0; i < s.ids.length; i++) {
      const distance = map.latLngToContainerPoint([s.lat[i], s.lon[i]])
        .distanceTo(event.containerPoint);
      if (distance <= bestDistance) { best = {s, i}; bestDistance = distance; }
    }
  }
  if (best === null) return;
  const {s, i} = best;
  L.popup().setLatLng([s.lat[i], s.lon[i]])
    .setContent(`Segment <b>${escapeHtml(s.ids[i])}</b><br>${escapeHtml(s.split)}<br>`
                + (s.selected[i] ? "In the selection" : "Not in the selection"))
    .openOn(map);
});

const bounds = L.latLngBounds(DATA.splits.flatMap(s => s.lat.map((lat, i) => [lat, s.lon[i]])));
if (bounds.isValid()) map.fitBounds(bounds, {padding: [20, 20]});
else map.setView([0, 0], 2);
</script>
</body>
</html>
"""
