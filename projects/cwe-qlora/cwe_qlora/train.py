import time
from pathlib import Path

from .compat import make_trainer, make_training_args, pick_optimizer, prepare_for_kbit
from .io_utils import read_json, read_jsonl, write_json
from .modeling import DEFAULT_MODEL, environment_info, load_base_model, resolve_dtype
from .tokenization import PadCollator, check_label_roundtrip, encode_dataset

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_arguments(parser) -> None:
    parser.add_argument("--work", default="work")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="outputs/adapter")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--resume", action="store_true")


class TokenizedDataset:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def run(args) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    work = Path(args.work)
    adapter_dir = Path(args.output)
    checkpoint_dir = adapter_dir.parent / "checkpoints"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = read_jsonl(work / "train.jsonl")
    prepare_meta = read_json(work / "prepare_meta.json") if (work / "prepare_meta.json").exists() else {}
    check_label_roundtrip(tokenizer, [row["label"] for row in train_rows])
    train_examples, dropped_train = encode_dataset(tokenizer, train_rows, args.max_length)
    val_examples, dropped_val = encode_dataset(tokenizer, read_jsonl(work / "val.jsonl"), args.max_length)
    if not train_examples:
        raise ValueError("No training examples after tokenization")

    quantize = not args.no_quant
    dtype = resolve_dtype(torch, args.dtype)
    model = load_base_model(args.model, quantize, dtype)
    model.config.use_cache = False
    if quantize:
        model = prepare_for_kbit(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        ),
    )
    trainable, total = count_parameters(model)

    training_args = make_training_args(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_examples else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim=pick_optimizer(quantize),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = make_trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(train_examples),
        eval_dataset=TokenizedDataset(val_examples) if val_examples else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
    )
    resume = args.resume and any(checkpoint_dir.glob("checkpoint-*"))
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=True) if resume else trainer.train()
    elapsed = time.perf_counter() - started

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    history = trainer.state.log_history
    eval_losses = [item["eval_loss"] for item in history if "eval_loss" in item]
    write_json(adapter_dir.parent / "training_log.json", history)
    environment = environment_info()
    write_json(
        work / "train_run.json",
        {
            "model": args.model,
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "dropped_train_too_long": dropped_train,
            "dropped_validation_too_long": dropped_val,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "quantized": quantize,
            "precision": str(dtype).replace("torch.", ""),
            "optimizer": training_args.optim if hasattr(training_args, "optim") else None,
            "seed": args.seed,
            "train_seconds": round(elapsed, 1),
            "train_loss": result.training_loss,
            "last_eval_loss": eval_losses[-1] if eval_losses else None,
            "global_steps": trainer.state.global_step,
            "device": environment["device"],
            "versions": {key: environment[key] for key in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
            "prepare_meta_present": (work / "prepare_meta.json").exists(),
            "run_mode": prepare_meta.get("run_mode", "unknown"),
            "data_fingerprint": prepare_meta.get("data_fingerprint"),
        },
    )
    print(f"adapter saved to {adapter_dir}")
    return 0
