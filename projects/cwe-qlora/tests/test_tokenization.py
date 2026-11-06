import pytest

from cwe_qlora.tokenization import check_label_roundtrip


class FakeTokenizer:
    def __init__(self, drop_prefix: bool):
        self.drop_prefix = drop_prefix

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [text]}

    def decode(self, ids, skip_special_tokens=True):
        text = ids[0]
        return text.removeprefix("CWE") if self.drop_prefix else text


def test_roundtrip_check_accepts_faithful_tokenizers():
    check_label_roundtrip(FakeTokenizer(drop_prefix=False), ["CWE-79", "CWE-22", "CWE-79"])


def test_roundtrip_check_rejects_tokenizers_that_lose_characters():
    with pytest.raises(ValueError, match="round-trip 2 label"):
        check_label_roundtrip(FakeTokenizer(drop_prefix=True), ["CWE-79", "CWE-22"])
