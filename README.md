# iRAP tools

Tools and pipelines built for the iRAP Vietnam road-survey work.

## Data loading, evaluation and analysis

| Package | Purpose |
|---|---|
| [`irap_data`](packages/irap_data/) | iRAP-BH and iRAP-Vietnam metadata (torch-free), dataset loaders, a dataset viewer (CLI `irap-dataset-viewer`) and a dataset statistics report (CLI `irap-dataset-stats`). |
| [`irap_evaluation`](packages/irap_evaluation/) | Evaluation, ensembling, coding-table export for saved model predictions. |
| [`irap_evaluation_server`](packages/irap_evaluation_server/) | Web app for an archive of prediction files with scoring, analysis and ensembling (CLI `irap-eval-server`). |

## Dataset preparation

| Package | Purpose |
|---|---|
| [`irap_video_cutting`](packages/irap_video_cutting/) | Cuts MP4 + GPX / WebGIS sidecar files at manual timestamps (CLI + GUI). |
| [`irap_vietnam_360`](packages/irap_vietnam_360/) | Extracts approximate perspective-projection images from Insta360 fisheye video using GPS tracks. **Note:** Superseded by [modified Gyroflow](https://github.com/Ivan1248/gyroflow). |
| [`irap_vietnam_data_preparation`](packages/irap_vietnam_data_preparation/) | Pipeline that turns extracted frames corresponding to road segments into a dataset compatible with the iRAP-BH dataset. |

Some parts of the dataset preparation pipeline were manual, and some automated parts implemented by other project participants are missing.
