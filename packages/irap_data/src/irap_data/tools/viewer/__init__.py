"""The `irap-dataset-viewer` command, which runs the Streamlit dataset viewer (`app`).

Needs the `viewer` extra. The command-line arguments are passed to `streamlit run`.
"""

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    streamlit_cli.main()  # Exits the process.
