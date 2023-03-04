"""Main entry point for `elk`."""

from .extraction import extract, ExtractionConfig
from .training import RunConfig
from .training.train import train
from pathlib import Path
from simple_parsing import ArgumentParser


def run():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract hidden states from a model.",
    )
    extract_parser.add_arguments(ExtractionConfig, dest="extraction")
    extract_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save hidden states to.",
        required=True,
    )
    extract_parser.add_argument(
        "--max_gpus",
        type=int,
        help="Maximum number of GPUs to use.",
        required=False,
        default=-1,
    )

    elicit_parser = subparsers.add_parser(
        "elicit",
        help=(
            "Extract and train a set of ELK reporters "
            "on hidden states from `elk extract`. "
        ),
        conflict_handler="resolve",
    )
    elicit_parser.add_arguments(RunConfig, dest="run")
    elicit_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save checkpoints to.",
    )

    subparsers.add_parser(
        "eval", help="Evaluate a set of ELK reporters generated by `elk train`."
    )
    args = parser.parse_args()

    if args.command == "extract":
        extract(args.extraction, args.max_gpus).save_to_disk(args.output)
    elif args.command == "elicit":
        train(args.run, args.output)

    elif args.command == "eval":
        # TODO: Implement evaluation script
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    run()