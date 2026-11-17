"""The `irap-dataset-stats` command: class frequencies and other statistics of an iRAP release.

The statistics are computed from the metadata only (see `irap_data.reports.statistics`), and the
reports are written by `irap_data.reports.statistics_report`. Run
`irap-dataset-stats <command> --help` for the options of each command.
"""

import argparse
import sys
import typing as T
from datetime import date
from pathlib import Path

from irap_data.metadata import DATASET_PRESETS, get_default_irap_paths, load_irap_metadata
from irap_data.reports.statistics import (
    ATTRIBUTE_ORDERS,
    CLASS_ORDERS,
    SELECTION_DESCRIPTIONS,
    compute_dataset_statistics,
)
from irap_data.reports.statistics_report import (
    PlotFormat,
    write_split_class_frequency_plot,
    write_statistics_reports,
)


# Commands #########################################################################################

def _get_metadata_dir(args: argparse.Namespace) -> Path:
    if args.metadata_dir is not None:
        return args.metadata_dir
    try:
        return get_default_irap_paths(args.dataset)[1]
    except RuntimeError as e:
        raise ValueError(f"{e} Alternatively, pass --metadata-dir.") from e


def run_report(args: argparse.Namespace) -> None:
    metadata = load_irap_metadata(_get_metadata_dir(args))
    statistics = compute_dataset_statistics(metadata, args.dataset, splits=args.splits,
                                            model_context_offsets=args.context_offsets)
    if statistics.is_model_selection_same_as_with_context:
        print("The model context offsets select the same segments as the context offsets of"
              " the release, so the model report would repeat the with-context report and is"
              " not written.")
    written = write_statistics_reports(
        statistics, args.out, report_date=date.today().isoformat(),
        metadata=metadata if args.map else None,
        plot_format=args.plot_format if args.plots else None, rare_fraction=args.rare_fraction,
        dpi=args.dpi)
    print("\n".join(f"Wrote {p}." for p in written))


def run_plot(args: argparse.Namespace) -> None:
    metadata = load_irap_metadata(_get_metadata_dir(args))
    write_split_class_frequency_plot(
        args.output, metadata, args.dataset, args.split, selection=args.selection,
        context_offsets=args.context_offsets, attribute_order=args.sort_attributes,
        class_order=args.sort_classes, normalize=args.normalize, log=not args.linear,
        dpi=args.dpi)
    print(f"Wrote {args.output}.")


# Argument parsing #################################################################################

def _parse_context_offsets(text: str) -> tuple[int, ...]:
    return tuple(int(o) for o in text.split(","))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", choices=list(DATASET_PRESETS), help="The release.")
    parser.add_argument("--metadata-dir", type=Path, default=None,
                        help="Default: the metadata directory of the release under $IRAP_HOME or"
                             " $DATASETS_PATH.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="The resolution of raster plots. Default: 200.")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irap-dataset-stats",
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser(
        "report", help="Write an HTML report and a CSV table of the class frequencies of each"
                       " selection of segments, with plots and a segment map, and a JSON"
                       " document of all statistics.")
    _add_common_arguments(report)
    report.add_argument("-o", "--out", type=Path, required=True, help="Output directory.")
    report.add_argument("--splits", nargs="+", default=None,
                        help="Default: all splits in splits.json. The first split with segments"
                             " is the base of the comparisons.")
    report.add_argument("--context-offsets", type=_parse_context_offsets, default=None,
                        metavar="OFFSETS",
                        help="The context offsets of a model, e.g. 0,-1,-4, for an additional"
                             " 'model' selection.")
    report.add_argument("--rare-fraction", type=float, default=0.01,
                        help="A class is rare below this share of the labeled segments of its"
                             " attribute. Default: 0.01.")
    report.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Leave out the plots, which need matplotlib.")
    report.add_argument("--no-map", dest="map", action="store_false",
                        help="Leave out the segment map of each report, an HTML page that loads"
                             " its basemap and the Leaflet library from the web.")
    report.add_argument("--plot-format", choices=T.get_args(PlotFormat), default="svg",
                        help="The file format of the plots. A PDF plot is linked from the report"
                             " and the others are embedded. Default: svg.")
    report.set_defaults(run=run_report)

    plot = commands.add_parser("plot", help="Plot the class frequencies of one split.")
    _add_common_arguments(plot)
    plot.add_argument("-o", "--output", type=Path, required=True,
                      help="The image path. Its extension selects the format, e.g. .pdf or .png.")
    plot.add_argument("--split", default="train", help="Default: train.")
    plot.add_argument("--selection", choices=list(SELECTION_DESCRIPTIONS), default="labeled",
                      help="Default: labeled. 'model' needs --context-offsets.")
    plot.add_argument("--context-offsets", type=_parse_context_offsets, default=None,
                      metavar="OFFSETS", help="The context offsets of the 'model' selection.")
    plot.add_argument("--normalize", action="store_true",
                      help="Plot the share of the labeled segments of the attribute instead of"
                           " the number of segments.")
    plot.add_argument("--linear", action="store_true",
                      help="Use a linear value axis. Default: logarithmic, as the counts span"
                           " several orders of magnitude.")
    plot.add_argument("--sort-attributes", choices=ATTRIBUTE_ORDERS, default="schema",
                      help="The order of the attributes (see"
                           " irap_data.reports.statistics.ATTRIBUTE_ORDERS). Default: schema.")
    plot.add_argument("--sort-classes", choices=CLASS_ORDERS, default="schema",
                      help="The order of the classes of an attribute (see"
                           " irap_data.reports.statistics.CLASS_ORDERS). Default: schema.")
    plot.set_defaults(run=run_plot)
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
