"""The Streamlit app of the iRAP dataset viewer.

Needs the `viewer` extra. Run it with the `irap-dataset-viewer` command (see
`irap_data.tools.viewer`), or with ``streamlit run <path-to>/irap_data/tools/viewer/app.py``.
The file uses absolute imports, so that Streamlit can run it as a script.
"""

import dataclasses as dc
import random
from pathlib import Path
import typing as T

import numpy as np
import streamlit as st

from irap_data import IRAP_DATASET_FACTORIES
from irap_data.jitter import make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
from irap_data.metadata import compute_label_matrix, resolve_datasets_root
from irap_data.tools.viewer.vis_utils import (
    create_composite_view_strip,
    format_class_value,
    get_index_color,
    tensor_image_to_uint8_np,
)


# Constants ####################################################################

#: Release name (a key of `IRAP_DATASET_FACTORIES`) -> (dataset directory, metadata directory)
#: under the datasets directory.
_RELEASE_TO_SUBDIRS = {
    "bh": ("IRAP_BIH", "IRAP_BIH_METADATA"),
    "vietnam": ("IRAP_Vietnam", "IRAP_Vietnam"),
}

_DATASETS_DIR_HELP = (
    "Parent directory containing one or more iRAP datasets.\n\n"
    "Expected layout:\n"
    "- iRAP-BH:    <datasets_dir>/IRAP_BIH/ (data)\n"
    "              <datasets_dir>/IRAP_BIH_METADATA/ (metadata)\n"
    "- iRAP-Vietnam: <datasets_dir>/IRAP_Vietnam/ (data + metadata together)\n\n"
    "Default is read from $IRAP_HOME, then $DATASETS_PATH."
)

#: ColorJitter option label -> jitter preset, None for no jitter.
_JITTER_PRESETS = {"None": None, "Standard": JITTER_STANDARD, "Strong (semi-supervised)": JITTER_STRONG}

_NEIGHBOR_RADIUS = 8

_VIEWER_CSS = """
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stImage"] {
        border-radius: 0px !important;
        display: flex;
        justify-content: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 0px !important;
        max-height: 90vh !important;
        width: 100% !important;
        max-width: 100% !important;
        object-fit: contain;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    body, [data-testid="stAppViewContainer"] * {
        opacity: 1 !important;
        filter: none !important;
        animation: none !important;
        transition: none !important;
    }
    .attributes-text {
        line-height: 1.0;
        font-size: 70%;
        display: flex;
        align-items: center;
        margin-bottom: 2px;
    }
    .seq-square {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 1px;
        border-radius: 1px;
        opacity: 0.8;
    }
    .seq-container {
        display: inline-flex;
        align-items: center;
        margin: 0 4px;
    }
    .attr-key {
        width: 200px;
        min-width: 140px;
        display: inline-block;
        text-align: right;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
"""


# Configuration ################################################################


@dc.dataclass(frozen=True)
class ViewerConfig:
    """The sidebar settings that the loaded dataset depends on (pending or active).

    Attributes:
        dataset_name: The release name, a key of `IRAP_DATASET_FACTORIES`.
    """

    dataset_name: str
    ds_dir: Path
    md_dir: Path
    split: str
    context_offsets: tuple[int, ...]

    @property
    def load_key(self) -> tuple:
        """The settings as a tuple of strings and integers, a key for the Streamlit caches."""
        return (self.dataset_name, str(self.ds_dir), str(self.md_dir), self.split, self.context_offsets)


# Dataset detection and loading ################################################


def _detect_releases(datasets_dir: Path) -> list[str]:
    return [name for name, (ds_subdir, _) in _RELEASE_TO_SUBDIRS.items() if (datasets_dir / ds_subdir).is_dir()]


@st.cache_resource(max_entries=64)
def load_example(_ds, idx: int, cache_key: tuple) -> dict:
    """`ds[idx]`, cached so that revisits do not decode the images again.

    `cache_key` is the active load key, which identifies `_ds` (Streamlit does not hash
    arguments with a leading underscore). The cached example is shared, not copied, so callers
    do not modify it.
    """
    return _ds[idx]


@st.cache_resource(hash_funcs={Path: str})
def load_splits(dataset_name: str, dataset_dir: Path, metadata_dir: Path, context_offsets: tuple):
    """The datasets of all splits of a release, cached so that a split change does not rebuild them."""
    return IRAP_DATASET_FACTORIES[dataset_name](
        dataset_dir=dataset_dir, metadata_dir=metadata_dir, context_offsets=context_offsets
    )


# Filtering ####################################################################


@st.cache_resource(max_entries=4)
def get_label_matrix(_ds, cache_key: tuple) -> np.ndarray:
    """The `(N, A)` label matrix of `_ds` (see `compute_label_matrix`), cached per load key."""
    return compute_label_matrix(_ds.segment_id_to_labels, _ds.segment_ids, len(_ds.info.attr_to_value_to_class_idx))


def compute_filtered_indices(
    label_matrix: np.ndarray,
    attr_to_value_to_class_idx: T.Mapping[str, T.Mapping[str, int]],
    filters: T.Mapping[str, T.Collection[str]],
) -> list[int]:
    """The rows of `label_matrix` whose labels are in the per-attribute allowed values.

    An attribute without allowed values in `filters` is not constrained.
    """
    is_match = np.ones(len(label_matrix), dtype=bool)
    for attr_idx, (attr, value_to_class_idx) in enumerate(attr_to_value_to_class_idx.items()):
        if allowed_values := filters.get(attr):
            allowed_class_indices = [value_to_class_idx[v] for v in allowed_values]
            is_match &= np.isin(label_matrix[:, attr_idx], allowed_class_indices)
    return np.flatnonzero(is_match).tolist()


# Neighbor sequence display ####################################################


def get_context_rgb_relpaths(ds, center_sid: str) -> list[tuple[int, str | None]]:
    """(offset, relative RGB path or None) of the context frames, in `context_offsets` order."""
    out = []
    for offset, sid in zip(ds.context_offsets, ds.segment_id_to_context_ids[center_sid]):
        p = ds.seg_to_paths[sid].get("rgb")
        out.append((offset, p.relative_to(ds.root).as_posix() if p is not None else None))
    return out


def get_neighbor_labels(ds, center_sid: str, radius: int) -> dict[int, T.Sequence[int]]:
    """Offset -> labels of the segments within `radius` positions of a segment along its road.

    Neighbors without labels in `ds` (for example, from another split) are left out.
    """
    road_id, pos = ds.seq_index[center_sid]
    seq = ds.road_to_seq[road_id]
    neighbors = {}
    for offset in range(-radius, radius + 1):
        neighbor_pos = pos + offset
        if offset != 0 and 0 <= neighbor_pos < len(seq):
            labels = ds.segment_id_to_labels.get(seq[neighbor_pos])
            if labels is not None:
                neighbors[offset] = labels
    return neighbors


def generate_sequence_html(
    values: T.Sequence[str],
    neighbors: T.Mapping[int, T.Sequence[int]],
    attr_idx: int,
    offsets: T.Iterable[int],
) -> str:
    """HTML squares colored by the neighbors' class of an attribute, transparent without a neighbor.

    Args:
        values: The values of the attribute, in class-index order.
        neighbors: Offset -> labels (see `get_neighbor_labels`).
        attr_idx: The index of the attribute in the labels.
    """
    squares = []
    for offset in offsets:
        if offset in neighbors:
            class_idx = neighbors[offset][attr_idx]
            squares.append(
                f'<div class="seq-square" style="background-color:{get_index_color(class_idx)};"'
                f' title="{format_class_value(values, class_idx)}"></div>'
            )
        else:
            squares.append('<div class="seq-square" style="background-color:transparent;"></div>')
    return "".join(squares)


# Sidebar sections #############################################################


def _render_sidebar_config() -> ViewerConfig | None:
    """Renders sidebar input widgets and returns the pending configuration.

    Returns None if the inputs are invalid or incomplete (error already shown).
    """
    try:
        default_datasets_dir = str(resolve_datasets_root())
    except RuntimeError:
        default_datasets_dir = ""
    datasets_dir_input = st.sidebar.text_input(
        "Datasets directory",
        value=default_datasets_dir,
        help=_DATASETS_DIR_HELP,
    )
    if not datasets_dir_input:
        st.error("Set $IRAP_HOME or $DATASETS_PATH, or enter a path in the sidebar.")
        return None
    datasets_dir = Path(datasets_dir_input).expanduser()

    releases = _detect_releases(datasets_dir)
    if not releases:
        looked_for = ", ".join(ds_subdir for ds_subdir, _ in _RELEASE_TO_SUBDIRS.values())
        st.error(f"No datasets found under {datasets_dir}. Looked for: {looked_for}.")
        return None
    dataset_name = st.sidebar.selectbox("Dataset", releases, format_func=lambda r: _RELEASE_TO_SUBDIRS[r][0])
    ds_subdir, md_subdir = _RELEASE_TO_SUBDIRS[dataset_name]

    split = st.sidebar.selectbox("Split", ["train", "val", "test"])

    seq_input = st.sidebar.text_input("Context offsets (e.g., 0, -1, -4)", value="0, -1, -4")
    try:
        context_offsets = tuple(int(x.strip()) for x in seq_input.split(","))
    except ValueError:
        st.error("Invalid context offsets format. Use comma-separated integers.")
        return None

    return ViewerConfig(
        dataset_name=dataset_name,
        ds_dir=datasets_dir / ds_subdir,
        md_dir=datasets_dir / md_subdir,
        split=split,
        context_offsets=context_offsets,
    )


def _ensure_dataset_loaded(config: ViewerConfig) -> tuple[T.Any, ViewerConfig] | None:
    """Renders the load button and returns the loaded dataset with its configuration.

    The dataset is loaded or reloaded only on a click, not on every widget change, so pending
    settings can differ from the active ones.

    Returns:
        (dataset, active configuration), or None before the first load or for an empty dataset.
    """
    active_config = st.session_state.get("active_config")
    # Compares the load keys, as each rerun of the script defines a new `ViewerConfig` class, which
    # makes the dataclass equality with a stored configuration false.
    is_stale = active_config is not None and active_config.load_key != config.load_key

    if active_config is None or is_stale:
        load_label = "Load dataset" if active_config is None else "Reload (settings changed)"
        if st.sidebar.button(load_label, width="stretch", type="primary"):
            st.session_state.active_config = active_config = config
            st.session_state.cursor = 0
            is_stale = False

    if active_config is None:
        st.info("Configure the sidebar settings and click **Load dataset** to begin.")
        return None

    if is_stale:
        st.sidebar.warning("Settings changed – showing previously loaded data. Click Reload to apply.")

    with st.spinner(f"Loading {active_config.ds_dir.name} / {active_config.split}..."):
        splits = load_splits(
            active_config.dataset_name,
            active_config.ds_dir,
            active_config.md_dir,
            active_config.context_offsets,
        )
    ds = splits[active_config.split]

    if len(ds) == 0:
        st.warning("Dataset is empty.")
        return None

    return ds, active_config


def _render_filters_and_navigation(
    ds,
    load_key: tuple,
    attr_values: dict[str, list[str]],
) -> int | None:
    """Renders filter widgets and navigation controls in the sidebar.

    Returns:
        The dataset index of the selected example, or None if no examples match.
    """
    # The load key in the widget keys gives each loaded dataset its own, initially empty, filters.
    filter_widget_keys = {a: f"filter_{hash(load_key)}_{a}" for a in attr_values}
    with st.sidebar.expander("Filter by attributes", expanded=False):
        if st.button("Clear filters", width="stretch"):
            for widget_key in filter_widget_keys.values():
                st.session_state[widget_key] = []
        filters = {a: st.multiselect(a, attr_values[a], key=widget_key) for a, widget_key in filter_widget_keys.items()}

    filtered = compute_filtered_indices(
        get_label_matrix(ds, cache_key=load_key), ds.info.attr_to_value_to_class_idx, filters
    )

    st.sidebar.write(f"Size: {len(ds)} segments, {len(filtered)} matching")

    if len(filtered) == 0:
        st.warning("No examples match the current filters.")
        return None

    # Wraps the index at the boundaries and converts negative indices to positive.
    st.session_state.cursor = st.session_state.get("cursor", 0) % len(filtered)

    if st.sidebar.button("Random", width="stretch"):
        st.session_state.cursor = random.randint(0, len(filtered) - 1)

    cursor = st.sidebar.number_input(
        "Index (within filtered)",
        min_value=-len(filtered),
        max_value=len(filtered),
        step=1,
        key="cursor",
    )
    ds_idx = filtered[cursor % len(filtered)]
    st.sidebar.caption(f"Dataset index: {ds_idx}")

    return ds_idx


# Main content columns #########################################################


def _render_image_column(data: dict, ds, jitter_selection: str) -> None:
    """Renders the image column: composite view strip with optional jitter preview."""
    rgb = data["rgb"]

    jitter_caption_suffix = ""
    if (jitter_preset := _JITTER_PRESETS[jitter_selection]) is not None:
        rgb = make_sequence_color_jitter(preset=jitter_preset)(dict(rgb=rgb))["rgb"]
        jitter_caption_suffix = f" [{jitter_selection.lower().split()[0]} jitter]"

    imgs_np = tensor_image_to_uint8_np(rgb)
    final_img = create_composite_view_strip(imgs_np)

    caption = f"Segment: {data['segment_id']}{jitter_caption_suffix}"
    if len(imgs_np) > 1:
        caption += f" | Context offsets: {', '.join(map(str, ds.context_offsets))}"

    st.image(final_img, caption=caption, width="stretch")

    lines = [
        f"{('+' if off > 0 else '')}{off}: {rel if rel is not None else '(missing)'}"
        for off, rel in get_context_rgb_relpaths(ds, data["segment_id"])
    ]
    st.markdown(
        "<div style='font-family:monospace;font-size:75%;line-height:1.2;"
        "white-space:nowrap;overflow-x:auto;'>" + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def _render_attribute_column(data: dict, ds, attr_values: dict[str, list[str]]) -> None:
    """Renders the attribute column: decoded labels with neighbor sequence squares."""
    neighbors = get_neighbor_labels(ds, data["segment_id"], radius=_NEIGHBOR_RADIUS)

    html_lines = []
    for attr_idx, (k, v_idx) in enumerate(zip(attr_values, map(int, data["target"]), strict=True)):
        v_str = format_class_value(attr_values[k], v_idx)
        color = get_index_color(v_idx)
        pre_html = generate_sequence_html(attr_values[k], neighbors, attr_idx, range(-_NEIGHBOR_RADIUS, 0))
        post_html = generate_sequence_html(attr_values[k], neighbors, attr_idx, range(1, _NEIGHBOR_RADIUS + 1))
        html_lines.append(
            f"""
            <div class="attributes-text">
                <span class="attr-key" title="{k}"><b>{k}</b>:</span>
                <span class="seq-container">{pre_html}</span>
                <span style="color:{color}; font-weight:semibold; margin: 0 4px;">{v_str}</span>
                <span class="seq-container">{post_html}</span>
            </div>
            """
        )
    st.markdown("".join(html_lines), unsafe_allow_html=True)


# Main #########################################################################


def main():
    st.set_page_config(layout="wide", page_title="iRAP Dataset Viewer")

    config = _render_sidebar_config()
    if config is None:
        return

    result = _ensure_dataset_loaded(config)
    if result is None:
        return
    ds, active_config = result

    # Attributes and their values, in class-index order.
    attr_values = {a: list(v_to_idx) for a, v_to_idx in ds.info.attr_to_value_to_class_idx.items()}

    ds_idx = _render_filters_and_navigation(ds, active_config.load_key, attr_values)
    if ds_idx is None:
        return

    try:
        data = load_example(ds, ds_idx, cache_key=active_config.load_key)
    except Exception as e:
        st.error(f"Error reading item {ds_idx}: {e}")
        return

    # Rendered after the navigation widgets, so that it is at the bottom of the sidebar.
    jitter_selection = st.sidebar.selectbox(
        "ColorJitter",
        list(_JITTER_PRESETS),
        help="Preview ColorJitter augmentation as used during training.",
    )

    st.markdown(_VIEWER_CSS, unsafe_allow_html=True)

    col_imgs, col_attrs = st.columns([1, 1])

    with col_imgs:
        _render_image_column(data, ds, jitter_selection)

    with col_attrs:
        _render_attribute_column(data, ds, attr_values)

    with st.expander("Raw Dictionary"):
        st.write({k: (str(v.shape) if hasattr(v, "shape") else v) for k, v in data.items()})


if __name__ == "__main__":
    main()
