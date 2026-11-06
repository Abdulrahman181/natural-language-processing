from .prompts import build_messages


def render_prompt(tokenizer, description: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(description), tokenize=False, add_generation_prompt=True
    )


def check_label_roundtrip(tokenizer, labels) -> None:
    broken = []
    for label in sorted(set(labels)):
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if decoded != label:
            broken.append(f"{label!r} -> {decoded!r}")
    if broken:
        shown = ", ".join(broken[:5])
        raise ValueError(
            f"Tokenizer does not round-trip {len(broken)} label(s), for example {shown}. "
            "Training or prediction on it would silently produce unparseable output."
        )


def encode_training_example(tokenizer, description: str, label: str, max_length: int) -> dict | None:
    prompt_ids = tokenizer(render_prompt(tokenizer, description), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    input_ids = list(prompt_ids) + list(answer_ids) + [eos_id]
    if len(input_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + list(answer_ids) + [eos_id]
    return {"input_ids": input_ids, "labels": labels}


def encode_dataset(tokenizer, rows: list[dict], max_length: int):
    examples = []
    dropped = 0
    for row in rows:
        encoded = encode_training_example(tokenizer, row["description"], row["label"], max_length)
        if encoded is None:
            dropped += 1
        else:
            examples.append(encoded)
    return examples, dropped


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        count = len(features)
        input_ids = torch.full((count, width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((count, width), -100, dtype=torch.long)
        attention_mask = torch.zeros((count, width), dtype=torch.long)
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            labels[index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
