"""The JavaScript libraries of the pages, downloaded into the data directory: those of the map of
the Analysis page, and MathJax, which typesets the formulas of the Methodology page.

The server serves them itself, since MapLibre starts its worker from the URL of its module, which
fails for a module of another origin, e.g. a CDN, and so that the pages do not depend on a CDN
while they are viewed. The pinned versions are checked by SHA-256.
"""

import dataclasses as dc
import hashlib
import typing as T
import urllib.request
from pathlib import Path

_MAPLIBRE_URL = "https://cdn.jsdelivr.net/npm/maplibre-gl@6.11.2/dist/"
_DECK_URL = "https://cdn.jsdelivr.net/npm/deck.gl@9.4.0/dist.min.js"


@dc.dataclass(frozen=True)
class LibraryFile:
    """
    Attributes:
        path: The path in the vendor directory, also of its URL on the server.
    """

    path: str
    url: str
    sha256: str


MAPLIBRE_MODULE = LibraryFile(
    "maplibre-gl/maplibre-gl.mjs", _MAPLIBRE_URL + "maplibre-gl.mjs",
    "3f55566295583644617fe17d008a36c580414b8c71dd2e1fcff1309de6fdee5d")
MAPLIBRE_STYLESHEET = LibraryFile(
    "maplibre-gl/maplibre-gl.css", _MAPLIBRE_URL + "maplibre-gl.css",
    "d8617d8421930e3fc6185365400e788c374c1a5d9fbe87999998c0bc14a202d3")
#: Sets the global `deck`.
DECK_SCRIPT = LibraryFile(
    "deck.gl/dist.min.js", _DECK_URL,
    "2eb6a1ae0d58604b1378682cd1136f8793478ba801e43dae48b3807e48758a6b")
MAP_LIBRARY_FILES = (
    MAPLIBRE_MODULE,
    # Imported by the module.
    LibraryFile("maplibre-gl/maplibre-gl-shared.mjs", _MAPLIBRE_URL + "maplibre-gl-shared.mjs",
                "76b5f55bdee928c65d592684aaff2b913d50b6b17b0ec6334e88b09b6aa47960"),
    LibraryFile("maplibre-gl/maplibre-gl-worker.mjs", _MAPLIBRE_URL + "maplibre-gl-worker.mjs",
                "01ad197aa7f4cec258a890febd71b7515e96309881b036a7095befc01a45296e"),
    MAPLIBRE_STYLESHEET,
    DECK_SCRIPT,
)

#: MathJax 3 with TeX input and SVG output, whose fonts it contains, so that it needs no other
#: files. MathJax 4 loads its fonts from separate files.
MATHJAX_SCRIPT = LibraryFile(
    "mathjax/tex-svg.js", "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js",
    "d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882")
MATH_LIBRARY_FILES = (MATHJAX_SCRIPT,)


def _fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def ensure_library_files(vendor_dir: Path, files: T.Iterable[LibraryFile],
                         fetch_url: T.Callable[[str], bytes] = _fetch_url) -> None:
    """Downloads the files that are missing in `vendor_dir` or differ from their SHA-256.

    Raises:
        OSError: If a download fails, e.g. without internet access.
        http.client.HTTPException: If a download breaks off, e.g. `IncompleteRead`.
        ValueError: If a downloaded file has another SHA-256.
    """
    for file in files:
        path = vendor_dir / file.path
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == file.sha256:
            continue
        data = fetch_url(file.url)
        if (sha256 := hashlib.sha256(data).hexdigest()) != file.sha256:
            raise ValueError(f"{file.url} has the SHA-256 {sha256}, not {file.sha256}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".download")
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
