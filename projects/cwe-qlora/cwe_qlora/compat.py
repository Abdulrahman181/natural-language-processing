import inspect


def make_training_args(**kwargs):
    from transformers import TrainingArguments

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = sorted(key for key in kwargs if key not in accepted)
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"TrainingArguments does not accept these options in this transformers version, skipped: {dropped}")
    return TrainingArguments(**kwargs)


def make_trainer(**kwargs):
    from transformers import Trainer

    tokenizer = kwargs.pop("tokenizer", None)
    try:
        return Trainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **kwargs)


def pick_optimizer(quantized: bool) -> str:
    if not quantized:
        return "adamw_torch"
    try:
        from transformers.training_args import OptimizerNames

        available = {item.value for item in OptimizerNames}
    except Exception:
        return "adamw_torch"
    for name in ("paged_adamw_8bit", "adamw_8bit", "adamw_torch"):
        if name in available:
            return name
    return "adamw_torch"


def prepare_for_kbit(model):
    from peft import prepare_model_for_kbit_training

    try:
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def load_pretrained_causal_lm(name: str, dtype, **kwargs):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
