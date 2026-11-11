# cwe-qlora: measured effect of QLoRA fine-tuning on CVE-to-CWE classification

A small, reproducible experiment: take a public instruction model, fine-tune it with 4-bit QLoRA to map a CVE description to a CWE identifier, and measure the result against the untuned model and against classical baselines on the same held-out rows.

This is a separate public-data experiment. It is not the client security-reports fine-tuning project and says nothing about it.

## What it measures

| System | What it is |
|---|---|
| `majority` | Always predicts the most frequent training label |
| `tfidf_svm_matched` | TF-IDF + linear SVM trained on the same subset the LLM sees |
| `tfidf_svm_full` | TF-IDF + linear SVM trained on the full training split |
| `base_zero_shot` | Qwen2.5-1.5B-Instruct, no adapter, same prompt |
| `qlora_tuned` | Same model + the trained LoRA adapter |

Every system is scored on the identical test rows. The LLM systems are evaluated by actual generation (greedy decoding), the output is parsed to `CWE-<n>`, and anything unparseable is counted as wrong and reported as an invalid-output rate.

Reported: accuracy, macro-F1 (over gold classes), invalid-output rate, bootstrap 95% confidence intervals, paired bootstrap differences between systems, and the same metrics on a "novel" subset of test rows whose description is not a near-duplicate (TF-IDF cosine >= 0.85) of anything in the full training split.

## Run it

Colab or any machine with a T4-class GPU:

1. Upload the project zip, open `notebooks/cwe_qlora_colab.ipynb`, set the runtime to T4 GPU, run the cells in order.
2. Or, on a GPU machine: `bash run_all.sh`.

Both start with a 20-step dry run on tiny data. Its numbers are meaningless; it only proves 4-bit loading, fp16 training and generation work on that machine. The dry run is explicitly tagged `run_mode=dry-run`, and the report command refuses to create a publishable report from it.

Manual commands:

```
python -m cwe_qlora prepare --out work
python -m cwe_qlora baselines --work work --out results
python -m cwe_qlora train --work work --output outputs/adapter
python -m cwe_qlora predict --work work --out-dir results
python -m cwe_qlora predict --work work --out-dir results --adapter outputs/adapter
python -m cwe_qlora report --work work --out results
```

For a Hub model, the pipeline resolves the model name to an immutable commit SHA before loading it. You can provide one explicitly when reproducing a published run:

```
python -m cwe_qlora train --work work --model-revision <model-commit-sha> --output outputs/adapter
python -m cwe_qlora predict --work work --model-revision <model-commit-sha> --out-dir results
```

For a diagnostic-only local run, mark the prepared artifacts explicitly:

```
python -m cwe_qlora prepare --out work_dry --run-mode dry-run --train-size 200 --val-size 50 --test-size 50
```

`report` rejects those artifacts by default. `--allow-dry-run` exists only for debugging and must never be used for a published metric.

Defaults: 12,000 training rows, 500 validation, 2,000 test, seed 13, 1 epoch, effective batch 32, learning rate 2e-4 with cosine schedule, LoRA r=16 / alpha=32 / dropout 0.05 on q,k,v,o,gate,up,down projections, max length 512.

Interrupted training can continue with `--resume`.

## Serving

The reusable `CWEInferenceEngine` is shared by the CLI and the optional FastAPI service. Install the serving extras and start the API locally with:

```
python -m pip install -e '.[serve]'
python -m cwe_qlora.api
```

The service exposes `/healthz`, `/readyz`, `/v1/metadata`, `/v1/predict`, and `/v1/predict:batch`. It lazy-loads the model, validates request sizes, uses deterministic decoding, and returns `503` when the inference backend is unavailable. A non-root Docker image is included; set `CWE_QLORA_HOST=0.0.0.0` only in a container or trusted network boundary.

## Engineering checks

The repository CI runs Python syntax and notebook validation, Ruff linting, Bandit security scanning, the CPU test suite, and a coverage gate. The current local verification passes 45 tests, Bandit, Ruff, and a 74% coverage threshold. The optional deep-learning integration tests remain skipped unless PyTorch, Transformers, PEFT, and Tokenizers are installed.

## Files to send back

`cwe_qlora_results.zip`, produced by the last notebook cell or by `run_all.sh`. It contains:

- `results/results.md` and `results/results.json`: the report
- `results/pred_*.jsonl` and `results/pred_*.meta.json`: per-row outputs and run metadata
- `work/prepare_meta.json`: dataset revision, data fingerprint, sizes, and leakage statistics
- `work/train_run.json`: hyperparameters, library versions, device, training time, final losses
- `outputs/training_log.json`: loss curve

Model weights are not needed.

## Read the numbers correctly

- Validation and test rows in `xamxte/cve-to-cwe` are filtered to rows where the NVD label equals an LLM-assigned label, so they are easier than raw CVE data. Training labels were also relabeled by an LLM. The score therefore measures agreement with those labels, not real-world accuracy.
- The split is random, not temporal.
- One seed, one run. Differences smaller than the reported confidence intervals are not evidence of a difference.
- The novel-subset numbers are the more conservative view and should be quoted next to the headline number.
- Training and inference time are measured from the run and recorded in `train_run.json`; any estimate given before the run is only an estimate.

## What was and was not verified before shipping

The CPU test suite covers prompt and label masking, tokenizer round-trip guard, data preparation and leakage checks, metrics, baselines, report generation, dry-run provenance enforcement, duplicate/unexpected prediction detection, data fingerprinting, and the complete prepare/train/predict/report path when the optional deep-learning dependencies are available. In the current verification, `pytest -q` passed 45 tests and skipped the optional training integration module because PyTorch/Transformers/PEFT were not installed in the isolated CPU environment.

Not verified here: 4-bit bitsandbytes loading, fp16 mixed precision on a T4, real-model memory use and speed, and the actual dataset download. The Colab dry run exists to expose any of these quickly; no performance number from it is valid for publication.

## Layout

```
cwe_qlora/      package: data, prompts, tokenization, train, predict, baselines, metrics, report
notebooks/      Colab notebook
tests/          CPU test suite
run_all.sh      same flow as the notebook for non-Colab GPU machines
```
