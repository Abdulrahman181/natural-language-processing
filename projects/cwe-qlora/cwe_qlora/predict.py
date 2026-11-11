import time
from pathlib import Path

from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .parsing import parse_prediction
from .service import CWEInferenceEngine, InferenceConfig


def add_arguments(parser) -> None:
    parser.add_argument("--work", default="work")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default=None, help="Hub commit SHA; resolved automatically when omitted.")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-quant", action="store_true")


def run(args) -> int:
    work = Path(args.work)
    rows = read_jsonl(work / "test.jsonl")
    prepare_meta = read_json(work / "prepare_meta.json") if (work / "prepare_meta.json").exists() else {}
    if args.limit:
        rows = rows[: args.limit]
    name = args.name or ("qlora_tuned" if args.adapter else "base_zero_shot")
    engine = CWEInferenceEngine(
        InferenceConfig(
            model=args.model,
            model_revision=args.model_revision,
            adapter=args.adapter,
            quantized=not args.no_quant,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    )
    started = time.perf_counter()
    raw_outputs = engine.predict_raw([row["description"] for row in rows])
    elapsed = time.perf_counter() - started
    records = [
        {"id": row["id"], "gold": row["label"], "raw": raw, "pred": parse_prediction(raw)}
        for row, raw in zip(rows, raw_outputs, strict=True)
    ]
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / f"pred_{name}.jsonl", records)
    write_json(
        out_dir / f"pred_{name}.meta.json",
        {
            "name": name,
            "model": args.model,
            "model_revision": engine.revision,
            "adapter": args.adapter,
            "rows": len(rows),
            "seconds": round(elapsed, 1),
            "rows_per_second": round(len(rows) / elapsed, 2) if elapsed else None,
            "batch_size": args.batch_size,
            "quantized": not args.no_quant,
            "precision": args.dtype,
            "device": engine.metadata()["environment"]["device"] if engine.loaded else None,
            "run_mode": prepare_meta.get("run_mode", "unknown"),
            "data_fingerprint": prepare_meta.get("data_fingerprint"),
        },
    )
    correct = sum(1 for record in records if record["gold"] == record["pred"])
    invalid = sum(1 for record in records if record["pred"] == "INVALID")
    print(f"{name}: accuracy {correct / len(records):.4f} on {len(records)} rows, invalid outputs {invalid}")
    return 0
