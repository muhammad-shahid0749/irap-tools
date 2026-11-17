"""The URL paths of the pages and endpoints. The names in braces are path parameters."""

import typing as T
import urllib.parse

from ..vendor_libraries import LibraryFile

SCORES_PATH = "/scores"
#: Its query parameters are those of `analysis_view.AnalysisView.to_path`.
ANALYSIS_PATH = "/analysis"
#: With the query parameters `dataset` and `models` (the regular expression of the shown model
#: labels).
MODELS_PATH = "/models"
#: With the query parameters `split` (of the scores) and an attribute subset
#: (`ATTRIBUTE_PARAMETER`).
MODEL_PATH = "/models/{model_id}"
#: With the query parameters `dataset`, `method` (the method name, not the display name), `split`
#: (of the scores) and an attribute subset (`ATTRIBUTE_PARAMETER`). A method has no number, and
#: its name can have any characters.
METHOD_PATH = "/methods"
#: How the scores are computed and how to read them (`methodology`).
METHODOLOGY_PATH = "/methodology"
ACTION_LOG_PATH = "/actions"
#: With the query parameters `next` (the path to return to) and `failed` ('1' after a wrong name
#: or password).
SIGN_IN_PATH = "/sign-in"
#: With the query parameter `next`.
REGISTER_PATH = "/register"
ACCOUNTS_PATH = "/accounts"
#: Where admins choose the shown metrics (`display_settings`).
SHOWN_METRICS_PATH = "/shown-metrics"
#: Endpoints of plain forms, with the form field `next`.
SIGN_IN_ENDPOINT_PATH = "/api/sign-in"
REGISTER_ENDPOINT_PATH = "/api/register"
SIGN_OUT_ENDPOINT_PATH = "/api/sign-out"
UPLOAD_PATH = "/api/uploads"
PREDICTIONS_DOWNLOAD_PATH = "/api/submissions/{submission_id}/predictions.parquet"
#: The files of a model (`ModelArchive.write_model_predictions_zip`).
MODEL_PREDICTIONS_DOWNLOAD_PATH = "/api/models/{model_id}/predictions.zip"
SCORES_DOWNLOAD_PATH = "/api/submissions/{submission_id}/scores/{evaluation_set}.json"
#: A coding table that a Model page prepared, which can be downloaded once.
CODING_TABLE_DOWNLOAD_PATH = "/api/coding-tables/{token}"
SEGMENT_IMAGE_PATH = "/api/datasets/{dataset}/images/{segment_id}"
#: The static files of the libraries (`vendor_libraries`).
VENDOR_PATH = "/vendor"

#: The query parameter of an attribute subset, repeated for each attribute.
ATTRIBUTE_PARAMETER = "attribute"
#: The query parameter of the order of the attribute rows of a table
#: (`score_tables.make_per_attribute_table_html`, `make_model_scores_html`,
#: `make_method_scores_html`).
ATTRIBUTE_SORT_PARAMETER = "attribute_sort"


def make_query_path(path: str, query: T.Mapping[str, str | T.Sequence[str]]) -> str:
    """Adds the query parameters that are not empty to `path`, a sequence as repeated
    parameters."""
    query = {k: v for k, v in query.items() if v}
    return f"{path}?{urllib.parse.urlencode(query, doseq=True)}" if query else path


def to_local_path(url: str, default: str) -> str:
    """Returns `url` if it is a path on this server, e.g. a `next` parameter, otherwise
    `default`, so that a link cannot redirect to another site."""
    parts = urllib.parse.urlsplit(url)
    # Browsers read '/\' as '//', the start of another host, and remove tabs and line breaks.
    is_local = (not parts.scheme and not parts.netloc and url.startswith("/")
                and not url.startswith(("//", "/\\")) and url.isprintable())
    return url if is_local else default


def get_sign_in_path(next_path: str) -> str:
    """Makes the path of the Sign-in page, which returns to `next_path`."""
    return make_query_path(SIGN_IN_PATH, {"next": next_path})


def get_register_path(next_path: str) -> str:
    """Makes the path of the Register page, which returns to `next_path`."""
    return make_query_path(REGISTER_PATH, {"next": next_path})


def get_models_path(dataset: str = "", model_pattern: str = "") -> str:
    """Makes the path of the Models page, with the shown dataset ('' for all) and the regular
    expression of the shown model labels ('' for all)."""
    return make_query_path(MODELS_PATH, {"dataset": dataset, "models": model_pattern})


def get_model_path(model_id: int, split: str = "", attributes: T.Sequence[str] = (),
                   attribute_sort: str = "") -> str:
    """Makes the path of the Model page, with the split of its scores, an attribute subset
    (`AttributeSubset.to_query`) and the order of the attribute rows."""
    return make_query_path(MODEL_PATH.format(model_id=model_id),
                           {"split": split, ATTRIBUTE_PARAMETER: attributes,
                            ATTRIBUTE_SORT_PARAMETER: attribute_sort})


def get_method_path(dataset: str, method_name: str, split: str = "",
                    attributes: T.Sequence[str] = (), attribute_sort: str = "") -> str:
    """Makes the path of the Method page, with the split of its scores, an attribute subset
    (`AttributeSubset.to_query`) and the order of the attribute rows."""
    return make_query_path(METHOD_PATH, {"dataset": dataset, "method": method_name,
                                         "split": split, ATTRIBUTE_PARAMETER: attributes,
                                         ATTRIBUTE_SORT_PARAMETER: attribute_sort})


def get_methodology_path(section_id: str) -> str:
    """Makes the path of a section of the Methodology page (`methodology.MethodologySection`)."""
    return f"{METHODOLOGY_PATH}#{section_id}"


def get_library_path(file: LibraryFile) -> str:
    """Makes the path of a file of a library (`vendor_libraries`) on the server."""
    return f"{VENDOR_PATH}/{file.path}"


def get_segment_image_path(dataset: str, segment_id: str) -> str:
    return SEGMENT_IMAGE_PATH.format(dataset=urllib.parse.quote(dataset, safe=""),
                                     segment_id=urllib.parse.quote(segment_id, safe=""))


def get_download_path(submission_id: int) -> str:
    return PREDICTIONS_DOWNLOAD_PATH.format(submission_id=submission_id)


def get_model_download_path(model_id: int) -> str:
    return MODEL_PREDICTIONS_DOWNLOAD_PATH.format(model_id=model_id)


def get_scores_download_path(submission_id: int, evaluation_set: str) -> str:
    return SCORES_DOWNLOAD_PATH.format(submission_id=submission_id, evaluation_set=evaluation_set)
