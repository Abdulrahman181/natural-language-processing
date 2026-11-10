import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

from helpers import make_rows, write_local_dataset  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from cwe_qlora.cli import main  # noqa: E402
from cwe_qlora.compat import make_training_args, pick_optimizer  # noqa: E402
from cwe_qlora.io_utils import read_json, read_jsonl  # noqa: E402
from cwe_qlora.prompts import SYSTEM_PROMPT  # noqa: E402
from cwe_qlora.tokenization import PadCollator, encode_dataset, encode_training_example, render_prompt  # noqa: E402

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def build_tokenizer():
    rows = make_rows(300, 5, 0)
    corpus = [row["description"] for row in rows] + sorted({row["cwe_id"] for row in rows})
    corpus += [SYSTEM_PROMPT, "system user assistant", "word " * 5]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    backend.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token=None
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny_model")
    tokenizer = build_tokenizer()
    tokenizer.save_pretrained(str(directory))
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
        initializer_range=0.3,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(str(directory))
    return directory


def test_training_refuses_a_tokenizer_that_cannot_round_trip_labels(tmp_path, tiny_model_dir):
    broken = tmp_path / "broken_model"
    shutil.copytree(tiny_model_dir, broken)
    word_level = Tokenizer(models.WordLevel(unk_token="<unk>"))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    word_level.train_from_iterator(["a b c 1 2 3 - "], trainers.WordLevelTrainer(special_tokens=["<unk>", *SPECIAL_TOKENS]))
    word_level.save(str(broken / "tokenizer.json"))
    raw = write_local_dataset(tmp_path / "raw", train=40, validation=10, test=10)
    work = tmp_path / "work"
    assert main(["prepare", "--source", "local", "--local-dir", str(raw), "--out", str(work),
                 "--train-size", "20", "--val-size", "5", "--test-size", "5", "--seed", "3"]) == 0
    with pytest.raises(ValueError, match="round-trip"):
        main(["train", "--work", str(work), "--model", str(broken), "--output", str(tmp_path / "out" / "adapter"),
              "--no-quant", "--max-steps", "1"])


def test_label_masking_keeps_prompt_out_of_the_loss():
    tokenizer = build_tokenizer()
    description = "Acme Portal 1.2 has a use-after-free in the file component"
    encoded = encode_training_example(tokenizer, description, "CWE-416", max_length=512)
    prompt_ids = tokenizer(render_prompt(tokenizer, description), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer("CWE-416", add_special_tokens=False)["input_ids"]
    assert encoded["input_ids"][: len(prompt_ids)] == prompt_ids
    assert encoded["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert encoded["labels"][len(prompt_ids):] == answer_ids + [tokenizer.eos_token_id]
    assert encoded["input_ids"][len(prompt_ids):] == answer_ids + [tokenizer.eos_token_id]
    assert prompt_ids[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")


def test_encode_dataset_drops_examples_that_exceed_max_length():
    tokenizer = build_tokenizer()
    rows = [{"description": "short", "label": "CWE-79"}, {"description": "word " * 400, "label": "CWE-79"}]
    examples, dropped = encode_dataset(tokenizer, rows, max_length=200)
    assert len(examples) == 1 and dropped == 1


def test_pad_collator_pads_and_masks():
    batch = PadCollator(pad_token_id=9)(
        [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 9, 9]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_compat_helpers_tolerate_unknown_options(tmp_path):
    arguments = make_training_args(output_dir=str(tmp_path), max_steps=1, definitely_not_an_option=True)
    assert arguments.max_steps == 1
    assert pick_optimizer(False) == "adamw_torch"
    assert isinstance(pick_optimizer(True), str)


def run_pipeline(tmp_path, model_dir, steps, learning_rate):
    raw = write_local_dataset(tmp_path / "raw", train=200, validation=30, test=40)
    work = tmp_path / "work"
    assert main(
        ["prepare", "--source", "local", "--local-dir", str(raw), "--out", str(work),
         "--train-size", "160", "--val-size", "16", "--test-size", "16", "--seed", "3"]
    ) == 0
    adapter = tmp_path / "out" / "adapter"
    assert main(
        ["train", "--work", str(work), "--model", str(model_dir), "--output", str(adapter), "--no-quant",
         "--max-steps", str(steps), "--batch-size", "8", "--grad-accum", "1", "--learning-rate", str(learning_rate),
         "--eval-steps", "100", "--save-steps", "1000", "--logging-steps", "1", "--max-length", "256",
         "--lora-r", "8", "--lora-alpha", "16"]
    ) == 0
    return work, adapter


def test_training_saves_an_adapter_and_run_metadata(tmp_path, tiny_model_dir):
    work, adapter = run_pipeline(tmp_path, tiny_model_dir, steps=4, learning_rate=1e-3)
    assert (adapter / "adapter_config.json").exists()
    run = read_json(work / "train_run.json")
    assert run["global_steps"] == 4
    assert run["quantized"] is False
    assert 0 < run["trainable_parameters"] < run["total_parameters"]
    assert run["train_examples"] == 160
    assert run["versions"]["torch"] and run["versions"]["peft"]
    assert (tmp_path / "out" / "training_log.json").exists()


def test_training_actually_reduces_the_loss(tmp_path, tiny_model_dir):
    _, _ = run_pipeline(tmp_path, tiny_model_dir, steps=60, learning_rate=3e-3)
    losses = [item["loss"] for item in read_json(tmp_path / "out" / "training_log.json") if "loss" in item]
    assert len(losses) >= 30
    assert sum(losses[-5:]) / 5 < 0.8 * sum(losses[:5]) / 5


def test_prediction_writes_parsed_outputs_for_base_and_adapter(tmp_path, tiny_model_dir):
    work, adapter = run_pipeline(tmp_path, tiny_model_dir, steps=3, learning_rate=1e-3)
    results = tmp_path / "results"
    common = ["--work", str(work), "--model", str(tiny_model_dir), "--out-dir", str(results),
              "--no-quant", "--limit", "10", "--batch-size", "4", "--max-new-tokens", "6"]
    assert main(["predict", *common]) == 0
    assert main(["predict", *common, "--adapter", str(adapter)]) == 0
    test_ids = [row["id"] for row in read_jsonl(work / "test.jsonl")][:10]
    for name in ("base_zero_shot", "qlora_tuned"):
        records = read_jsonl(results / f"pred_{name}.jsonl")
        assert [record["id"] for record in records] == test_ids
        assert all({"gold", "raw", "pred"} <= set(record) for record in records)
        assert read_json(results / f"pred_{name}.meta.json")["rows"] == 10


def test_overfit_model_recovers_labels_end_to_end(tmp_path, tiny_model_dir):
    raw = write_local_dataset(tmp_path / "raw", train=200, validation=30, test=40)
    work = tmp_path / "work"
    main(["prepare", "--source", "local", "--local-dir", str(raw), "--out", str(work),
          "--train-size", "200", "--val-size", "16", "--test-size", "40", "--seed", "3"])
    adapter = tmp_path / "out" / "adapter"
    main(["train", "--work", str(work), "--model", str(tiny_model_dir), "--output", str(adapter), "--no-quant",
          "--max-steps", "250", "--batch-size", "8", "--grad-accum", "1", "--learning-rate", "5e-3",
          "--eval-steps", "1000", "--save-steps", "1000", "--logging-steps", "50", "--max-length", "256",
          "--lora-r", "16", "--lora-alpha", "32"])
    results = tmp_path / "results"
    main(["predict", "--work", str(work), "--model", str(tiny_model_dir), "--adapter", str(adapter),
          "--out-dir", str(results), "--no-quant", "--batch-size", "8", "--max-new-tokens", "8"])
    records = read_jsonl(results / "pred_qlora_tuned.jsonl")
    accuracy = sum(1 for record in records if record["gold"] == record["pred"]) / len(records)
    assert accuracy > 0.6
