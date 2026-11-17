"""Find the images with equal file content, and the images to leave out because of them.

A group is the set of all segments with the same image, if there are 2 or more. Two images are
equal if their files have equal content (the same size and SHA-256 hash). Most files are not
read whole: the images are grouped by file size, then by a hash of their first bytes
(`HEAD_NUM_BYTES`), and only the images that still share a group by a hash of their whole
content.

A folder span holds the segments of an image folder, in seg_id order, from the first to the
last segment of a group. It can hold other segments, whose images differ. Its position says
where it lies in its folder: at the start, in the middle, at the end, or it is the whole
folder.

`find_images_to_exclude` decides which images to leave out (see its docstring), and
build_metadata.py leaves them out. find_repeated_images.py reports the groups.
"""

import dataclasses
import hashlib
import typing as T
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# The number of bytes at the start of a file that are hashed before whole files are.
HEAD_NUM_BYTES = 1 << 16
READ_CHUNK_NUM_BYTES = 1 << 20
# Where a run of segments lies in its sequence. A "whole" run is the whole sequence.
RUN_POSITIONS = ("start", "middle", "end", "whole")
# The least number of equal images at the start or end of an image folder that makes it
# frozen. The frames of such a folder were sampled with time-shifted GNSS coordinates, so
# that the positions outside the video got its first or last frame, and the other images are
# offset from their labels by about as many positions. The 2-image runs at the end of
# VID_20241210_172830_00_003_2 and VID_20241213_105626_00_051_2 are taken as a shift of 10 m,
# less than the 20 m labeling step, so only their repeated image is left out.
MIN_NUM_FROZEN_IMAGES = 3

K = T.TypeVar("K")


@dataclasses.dataclass(frozen=True)
class FolderSpan:
    """The segments of an image folder from the first to the last segment of a group."""
    folder: str
    group_index: int
    seg_ids: list[str]  # in seg_id order, with the segments that are not in the group
    num_group_segments: int
    position: str  # in the folder, one of `RUN_POSITIONS`

    @property
    def num_other_segments(self) -> int:
        return len(self.seg_ids) - self.num_group_segments


def get_run_position(start: int, num_run_segments: int, num_segments: int) -> str:
    """Returns where a run that starts at index `start` lies in its sequence."""
    is_at_start = start == 0
    is_at_end = start + num_run_segments == num_segments
    if is_at_start and is_at_end:
        return "whole"
    return "start" if is_at_start else "end" if is_at_end else "middle"


def hash_file_head(path: Path) -> bytes:
    with path.open("rb") as file:
        return hashlib.sha256(file.read(HEAD_NUM_BYTES)).digest()


def hash_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(READ_CHUNK_NUM_BYTES):
            digest.update(chunk)
    return digest.digest()


def split_groups(groups: list[list[K]], key: T.Callable[[K], T.Hashable], *, desc: str
                 ) -> list[list[K]]:
    """Splits each group into groups of items with equal keys, and drops the groups of 1 item."""
    result = []
    with tqdm(total=sum(map(len, groups)), desc=desc) as progress:
        for group in groups:
            key_to_items = defaultdict(list)
            for item in group:
                key_to_items[key(item)].append(item)
                progress.update()
            result.extend(items for items in key_to_items.values() if len(items) > 1)
    return result


def find_equal_file_groups(name_to_path: T.Mapping[str, Path]) -> list[list[str]]:
    """Finds the groups of 2 or more files with equal content and returns the names of each.

    Each step compares only the files that the cheaper steps before it could not tell apart.
    """
    groups = [list(name_to_path)]
    for desc, key in (
            ("Comparing file sizes", lambda name: name_to_path[name].stat().st_size),
            ("Comparing first bytes", lambda name: hash_file_head(name_to_path[name])),
            ("Comparing whole files", lambda name: hash_file(name_to_path[name]))):
        groups = split_groups(groups, key, desc=desc)
    return groups


def get_seg_id_to_group_index(groups: list[list[str]]) -> dict[str, int]:
    return {seg_id: i for i, group in enumerate(groups) for seg_id in group}


def get_folder_to_seg_ids(seg_id_to_image_path: T.Mapping[str, Path]) -> dict[str, list[str]]:
    """Returns the seg_ids of the images of each image folder, in seg_id order."""
    folder_to_seg_ids = defaultdict(list)
    for seg_id, path in seg_id_to_image_path.items():
        folder_to_seg_ids[path.parent.name].append(seg_id)
    return {folder: sorted(seg_ids, key=int) for folder, seg_ids in folder_to_seg_ids.items()}


def find_folder_spans(folder_to_seg_ids: T.Mapping[str, T.Sequence[str]],
                      groups: list[list[str]]) -> list[FolderSpan]:
    """Finds the folder spans of each group, in the order of the groups."""
    seg_id_to_group_index = get_seg_id_to_group_index(groups)
    group_index_to_spans = defaultdict(list)
    for folder, seg_ids in folder_to_seg_ids.items():
        group_index_to_indices = defaultdict(list)
        for i, seg_id in enumerate(seg_ids):
            if (group_index := seg_id_to_group_index.get(seg_id)) is not None:
                group_index_to_indices[group_index].append(i)
        for group_index, indices in group_index_to_indices.items():
            start, end = indices[0], indices[-1] + 1
            group_index_to_spans[group_index].append(FolderSpan(
                folder, group_index, list(seg_ids[start:end]), len(indices),
                get_run_position(start, end - start, len(seg_ids))))
    return [span for i in range(len(groups)) for span in group_index_to_spans[i]]


def is_frozen_span(span: FolderSpan) -> bool:
    """Says whether a folder span is a frozen frame at the start or end of its folder.

    Such a span holds at least `MIN_NUM_FROZEN_IMAGES` images of its group, and more than
    twice as many as other images, as a few frames inside a frozen run can differ.
    """
    return (span.position != "middle"
            and span.num_group_segments >= MIN_NUM_FROZEN_IMAGES
            and span.num_group_segments > 2 * span.num_other_segments)


def find_images_to_exclude(folder_to_seg_ids: T.Mapping[str, T.Sequence[str]],
                           groups: list[list[str]]
                           ) -> tuple[dict[str, str], dict[str, str]]:
    """Finds the image folders and the seg_ids to leave out because of equal images.

    1. A folder with a frozen frame at its start or end (`is_frozen_span`) is left out
       whole, as all its images are offset from their labels (see `MIN_NUM_FROZEN_IMAGES`).
    2. Of the other images of each group, all but the one with the lowest seg_id are left
       out. In the checked cases, it is the first one that fits its neighbours: in a short
       run at the end of a folder, the others repeat it, and elsewhere they are frames from
       the wrong time.

    Returns:
        ``{folder: reason}`` and ``{seg_id: reason}`` in the order of the seg_ids.
    """
    folder_to_reason = {}
    for span in find_folder_spans(folder_to_seg_ids, groups):
        if is_frozen_span(span) and span.folder not in folder_to_reason:
            folder_to_reason[span.folder] = (
                f"{span.num_group_segments} equal images of"
                f" {len(folder_to_seg_ids[span.folder])} at the {span.position} of the folder"
                f" ({span.seg_ids[0]}-{span.seg_ids[-1]}): the GNSS coordinates used to"
                f" sample the frames are probably time-shifted")
    seg_ids_in_excluded_folders = {s for folder in folder_to_reason
                                   for s in folder_to_seg_ids[folder]}
    seg_id_to_reason = {}
    for group in groups:
        seg_ids = sorted((s for s in group if s not in seg_ids_in_excluded_folders), key=int)
        for seg_id in seg_ids[1:]:
            seg_id_to_reason[seg_id] = f"equal to the image of {seg_ids[0]}"
    return folder_to_reason, dict(sorted(seg_id_to_reason.items(), key=lambda item: int(item[0])))
