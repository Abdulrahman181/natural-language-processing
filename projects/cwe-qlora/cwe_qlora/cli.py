import argparse
import sys
from pathlib import Path

from .baselines import run_baselines
from .data import (
    DEFAULT_DATASET,
    dataset_revision,
    load_hub_splits,
    load_local_splits,
    meta_to_json,
    prepare_splits,
)
from .io_utils import write_jsonl
from .report import DEFAULT_NOVEL_THRESHOLD, render_markdown, write_report


def command_prepare(args) -> int:
    if args.source == "local":
        if not args.local_dir:
            print("--local-dir is required with --source local")
            return 2
        raw = load_local_splits(args.local_dir)
        revision = None
        dataset = f"local:{args.local_dir}"
    else:
        raw = load_hub_splits(args.dataset, args.revision)
        revision = args.revision or dataset_revision(args.dataset)
        dataset = args.dataset
    splits, meta = prepare_splits(raw, args.train_size, args.val_size, args.test_size, args.seed, args.max_per_class)
    meta["dataset"] = dataset
    meta["revision"] = revision
    meta["run_mode"] = args.run_mode
    out = Path(args.out)
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    (out / "prepare_meta.json").write_text(meta_to_json(meta), encoding="utf-8")
    print(meta_to_json(meta))
    return 0


def command_baselines(args) -> int:
    meta = run_baselines(args.work, args.out)
    print(meta)
    return 0


def command_train(args) -> int:
    from .train import run

    return run(args)


def command_predict(args) -> int:
    from .predict import run

    return run(args)


def command_report(args) -> int:
    report = write_report(args.work, args.out, args.threshold, allow_dry_run=args.allow_dry_run)
    print(render_markdown(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import predict, train

    parser = argparse.ArgumentParser(prog="cwe_qlora")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", default="work")
    prepare.add_argument("--source", choices=["hub", "local"], default="hub")
    prepare.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare.add_argument("--revision", default=None)
    prepare.add_argument("--local-dir", default=None)
    prepare.add_argument("--train-size", type=int, default=12000)
    prepare.add_argument("--val-size", type=int, default=500)
    prepare.add_argument("--test-size", type=int, default=2000)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.add_argument("--max-per-class", type=int, default=None)
    prepare.add_argument(
        "--run-mode", choices=["full", "dry-run"], default="full",
        help="Provenance label; dry-run artifacts are rejected by report by default.",
    )
    prepare.set_defaults(func=command_prepare)

    baselines = commands.add_parser("baselines")
    baselines.add_argument("--work", default="work")
    baselines.add_argument("--out", default="results")
    baselines.set_defaults(func=command_baselines)

    trainer = commands.add_parser("train")
    train.add_arguments(trainer)
    trainer.set_defaults(func=command_train)

    predictor = commands.add_parser("predict")
    predict.add_arguments(predictor)
    predictor.set_defaults(func=command_predict)

    report = commands.add_parser("report")
    report.add_argument("--work", default="work")
    report.add_argument("--out", default="results")
    report.add_argument("--threshold", type=float, default=DEFAULT_NOVEL_THRESHOLD)
    report.add_argument(
        "--allow-dry-run", action="store_true",
        help="Explicitly allow a dry-run report; never use it for published metrics.",
    )
    report.set_defaults(func=command_report)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)
