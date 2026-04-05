import argparse
import runpy
import sys


COMMAND_MODULES = {
    "api": "gauge_read.api",
    "train": "gauge_read.train",
    "train-stn": "gauge_read.train_stn",
    "valid": "gauge_read.validation",
    "infer": "gauge_read.inference",
    "webui": "gauge_read.webui.webui",
    "gui": "gauge_read.webui.gui",
}

COMMAND_HELP = {
    "api": "Launch the FastAPI service",
    "train": "Train the main meter reading model",
    "train-stn": "Train the STN correction model",
    "valid": "Run validation on a labeled dataset",
    "infer": "Run single-image CLI inference",
    "webui": "Launch the Gradio WebUI",
    "gui": "Launch the desktop GUI wrapper",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gaugeread",
        description="Gauge Read unified command line entrypoint",
        usage="gaugeread <subcommand> [options]",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=tuple(COMMAND_MODULES.keys()),
        help="Available subcommands:\n" + "\n".join(f"  {name:<9} {COMMAND_HELP[name]}" for name in COMMAND_MODULES),
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    subcommand = argv[0]
    if subcommand not in COMMAND_MODULES:
        parser.error(f"invalid subcommand: {subcommand}")

    module_name = COMMAND_MODULES[subcommand]
    remaining = argv[1:]

    original_argv = sys.argv[:]
    try:
        sys.argv = [f"gaugeread {subcommand}", *remaining]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = original_argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
