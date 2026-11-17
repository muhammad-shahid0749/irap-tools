"""Standard subdirectory layout under an ``IRAP_Vietnam/`` dataset root.

All Vietnam pipeline scripts take a single positional ``<data_dir>`` argument (the
dataset root) and derive their inputs/outputs from these constants.
"""

from pathlib import Path

RAW_SUBDIR = "_raw"
WORK_SUBDIR = "_work"
IMAGES_SUBDIR = "FRAMES"
IMAGES_DUPLICATES_SUBDIR = "FRAMES_duplicates"
PATCH_ARCHIVES_TMP_SUBDIR = "patch_archives_tmp"

RARS_SUBDIR_UNDER_RAW = "image_rars"
ATTR_META_FILENAME = "attribute_metadata.json"
CODING_TABLES_ZIP = "coding-tables.zip"
CODING_TABLES_SUBDIR = "coding-tables"
ROWS_PARQUET = "rows.parquet"
PARSE_REPORT = "parse_report.json"
CROSS_ANNOTATOR_REPORT = "cross_annotator_report.json"
BUILD_REPORT = "build_report.json"
REPEATED_IMAGES_REPORT = "repeated_images_report.json"
UNLABELED_SEGMENT_IDS_FILENAME = "unlabeled_segment_ids.json"
UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME = "unlabeled_sequence_id_to_data.json"
UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME = "unlabeled_unlocated_segment_ids.json"
SEGMENT_ID_TO_DATA_PATHS_REL_FILENAME = "segment_id_to_data_paths_rel.json"
SEGMENT_ID_TO_ROAD_DATA_FILENAME = "segment_id_to_road_data.json"
ROAD_ID_TO_SEGMENT_ID_SEQUENCE_FILENAME = "road_id_to_segment_id_sequence.json"
SPLITS_FILENAME = "splits.json"


def raw_dir(data_dir: Path) -> Path:
    return data_dir / RAW_SUBDIR


def work_dir(data_dir: Path) -> Path:
    return data_dir / WORK_SUBDIR


def images_dir(data_dir: Path) -> Path:
    return data_dir / IMAGES_SUBDIR


def images_duplicates_dir(data_dir: Path) -> Path:
    return work_dir(data_dir) / IMAGES_DUPLICATES_SUBDIR


def patch_archives_tmp_dir(data_dir: Path) -> Path:
    return work_dir(data_dir) / PATCH_ARCHIVES_TMP_SUBDIR


def metadata_dir(data_dir: Path) -> Path:
    return data_dir


def rars_dir(data_dir: Path) -> Path:
    return raw_dir(data_dir) / RARS_SUBDIR_UNDER_RAW


def attr_meta_path(data_dir: Path) -> Path:
    return raw_dir(data_dir) / ATTR_META_FILENAME


def coding_tables_zip_path(data_dir: Path) -> Path:
    return raw_dir(data_dir) / CODING_TABLES_ZIP


def coding_tables_dir(data_dir: Path) -> Path:
    return work_dir(data_dir) / CODING_TABLES_SUBDIR


def rows_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / ROWS_PARQUET


def parse_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / PARSE_REPORT


def cross_annotator_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / CROSS_ANNOTATOR_REPORT


def build_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / BUILD_REPORT


def repeated_images_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / REPEATED_IMAGES_REPORT


def unlabeled_segment_ids_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_SEGMENT_IDS_FILENAME


def unlabeled_sequence_id_to_data_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME


def unlabeled_unlocated_segment_ids_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME


def output_attr_meta_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / ATTR_META_FILENAME


def segment_id_to_data_paths_rel_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SEGMENT_ID_TO_DATA_PATHS_REL_FILENAME


def segment_id_to_road_data_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SEGMENT_ID_TO_ROAD_DATA_FILENAME


def road_id_to_segment_id_sequence_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / ROAD_ID_TO_SEGMENT_ID_SEQUENCE_FILENAME


def splits_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SPLITS_FILENAME
