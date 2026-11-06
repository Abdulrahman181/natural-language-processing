import time
from pathlib import Path

from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .modeling import DEFAULT_MODEL, environment_info, load_base_model, resolve_dtype
from .parsing import parse_prediction
from .tokenization import check_label_roundtrip, render_prompt


def add_arguments(parser) -> None:
    parser.add_argument("--work", default="work")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-quant", action="store_true")


def run(args) -> int:
    import torch
    from transformers import AutoTokenizer

    work = Path(args.work)
    rows = read_jsonl(work / "test.jsonl")
    prepare_meta = read_json(work / "prepare_meta.json") if (work / "prepare_meta.json").exists() else {}
    if args.limit:
        rows = rows[: args.limit]
    name = args.name or ("qlora_tuned" if args.adapter else "base_zero_shot")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    check_label_roundtrip(tokenizer, [row["label"] for row in rows])
    dtype = resolve_dtype(torch, args.dtype)
    model = load_base_model(args.model, not args.no_quant, dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["description"]))
    outputs = [None] * len(rows)
    started = time.perf_counter()
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        prompts = [render_prompt(tokenizer, rows[index]["description"]) for index in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for index, text in zip(batch, texts):
            outputs[index] = text.strip()
    elapsed = time.perf_counter() - started

    records = [
        {"id": row["id"], "gold": row["label"], "raw": raw, "pred": parse_prediction(raw)}
        for row, raw in zip(rows, outputs)
    ]
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / f"pred_{name}.jsonl", records)
    environment = environment_info()
    write_json(
        out_dir / f"pred_{name}.meta.json",
        {
            "name": name,
            "model": args.model,
            "adapter": args.adapter,
            "rows": len(rows),
            "seconds": round(elapsed, 1),
            "rows_per_second": round(len(rows) / elapsed, 2) if elapsed else None,
            "batch_size": args.batch_size,
            "quantized": not args.no_quant,
            "precision": str(dtype).replace("torch.", ""),
            "device": environment["device"],
            "run_mode": prepare_meta.get("run_mode", "unknown"),
            "data_fingerprint": prepare_meta.get("data_fingerprint"),
        },
    )
    correct = sum(1 for record in records if record["gold"] == record["pred"])
    invalid = sum(1 for record in records if record["pred"] == "INVALID")
    print(f"{name}: accuracy {correct / len(records):.4f} on {len(records)} rows, invalid outputs {invalid}")
    return 0
