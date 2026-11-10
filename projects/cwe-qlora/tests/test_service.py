import pytest

from cwe_qlora.service import CWEInferenceEngine, InferenceConfig


def test_service_validates_configuration_and_inputs():
    with pytest.raises(ValueError, match="batch_size"):
        CWEInferenceEngine(InferenceConfig(batch_size=0))
    engine = CWEInferenceEngine(InferenceConfig())
    with pytest.raises(TypeError, match="descriptions"):
        engine.predict_raw("not-a-list")
    with pytest.raises(ValueError, match="must not be empty"):
        engine.predict_raw(["   "])


def test_empty_batch_is_safe_without_loading_a_model():
    engine = CWEInferenceEngine(InferenceConfig())
    assert engine.predict_raw([]) == []
    assert engine.loaded is False
    assert engine.metadata()["loaded"] is False
