from pathlib import Path

from .compat import load_pretrained_causal_lm

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_dtype(torch, preference: str = "auto"):
    if not torch.cuda.is_available():
        return torch.float32
    if preference == "auto":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return getattr(torch, preference)


def resolve_model_revision(name: str, requested: str | None = None) -> str | None:
    """Resolve a Hub model to an immutable commit while preserving local model paths."""
    if Path(name).exists():
        return requested
    if requested:
        return requested
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name).sha
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve an immutable revision for Hub model {name!r}. "
            "Pass --model-revision with a commit SHA or check network access."
        ) from exc


def load_base_model(name: str, quantize: bool, dtype, revision: str | None = None):
    import torch

    kwargs = {}
    if quantize:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading needs a CUDA GPU. Pass --no-quant to run without quantization.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    model = load_pretrained_causal_lm(name, dtype, revision=revision, **kwargs)
    if not quantize and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def environment_info() -> dict:
    import platform
    from importlib import metadata

    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "scikit-learn"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["cuda_available"] = False
        info["device"] = "cpu"
    return info
