from ballast.schema import (ALL_METRICS, BallastConfig, EngineConfig, Metric, MetricSet, ModelConfig, CorpusConfig, RuntimeConfig, PipelineConfig, SamplingMode, Task, LogLevel, CacheType)
from datetime import datetime
from pathlib import Path
import llama_cpp # type: ignore
import logging
import yaml
import uuid

def run_time():
    return datetime.now().astimezone()

repo_root = Path(__file__).resolve().parent.parent
ballast_yaml = repo_root / "ballast.yaml"
engines_dir = repo_root / "engines"
eval_dir = repo_root / "eval"
prompts_dir = repo_root / "eval" / "prompts"
perplexity_dir = repo_root / "eval" / "perplexity"
results_dir = repo_root / "results"
models_dir = repo_root / "models"
run_timestamp = (run_time()).strftime("%Y-%m-%d_%H-%M")
run_id = str(uuid.uuid4())
run_folder = results_dir / f"Benchmark_{run_timestamp}"

REQUIRED_BINARIES = ["llama-perplexity"]

KV_TYPE_MAP = {
    ct: getattr(llama_cpp, f"GGML_TYPE_{ct.name}")
    for ct in CacheType
    if hasattr(llama_cpp, f"GGML_TYPE_{ct.name}")
}

BYTES_PER_ELEM = {
    CacheType.F32: 4.0,
    CacheType.F16: 2.0,
    CacheType.BF16: 2.0,
    CacheType.Q8_0: 1.0,
    CacheType.Q5_0: 0.625,
    CacheType.Q5_1: 0.6875,
    CacheType.Q4_0: 0.5625,
    CacheType.Q4_1: 0.625,
    CacheType.IQ4_NL: 0.5625,
}

KV_ALLOWED = frozenset(KV_TYPE_MAP.keys())

def load_config() -> BallastConfig:

    if not ballast_yaml.exists():
        raise ValueError(
        f"\n> No ballast.yaml found at {ballast_yaml}"
        f"\n-> Create a ballast.yaml at the repo root before running."
        )
    
    with open(ballast_yaml) as f:
        raw =  yaml.safe_load(f)

    return BallastConfig(
        engines=[_engine_from_dict(e) for e in raw["engines"]],
        models=[_model_from_dict(m) for m in raw["models"]],
        corpora=[_corpus_from_dict(c) for c in raw["corpora"]],
        runtime=_runtime_from_dict(raw["runtime"]),
        pipeline=_pipeline_from_dict(raw["pipeline"]),
        metrics=_metrics_from_raw(raw["metrics"]),
    )


def _engine_from_dict(raw: dict) -> EngineConfig:
    return EngineConfig(
        name=raw["name"],
        source=raw.get("source"),
        tag=raw.get("tag"),
        cmake_flags=raw.get("cmake_flags", {}),
        path=raw.get("path"),
    )


def _model_from_dict(raw: dict) -> ModelConfig:
    return ModelConfig(
        name=raw["name"],
        source=raw["source"],
        context_size=raw["context_size"],
        cache_type_k=CacheType(raw["cache_type_k"]) if raw.get("cache_type_k") else None,
        cache_type_v=CacheType(raw["cache_type_v"]) if raw.get("cache_type_v") else None,
        fa=raw["fa"],
        batch_size=raw["batch_size"],
        generated_tokens=raw["generated_tokens"],
        swa_full=raw["swa_full"],
    )


def _corpus_from_dict(raw: dict) -> CorpusConfig:
    return CorpusConfig(
        name=raw["name"],
        source=raw["source"],
        chunks=raw["chunks"],
    )


def _runtime_from_dict(raw: dict) -> RuntimeConfig:
    return RuntimeConfig(
        n_threads=raw["n_threads"],
        n_threads_batch=raw["n_threads_batch"],
    )


def _pipeline_from_dict(raw: dict) -> PipelineConfig:
    return PipelineConfig(
        mode=SamplingMode(raw["mode"]),
        task=Task(raw["task"]),
        auto_repair=raw["auto_repair"],
        sample_interval_ms=raw["sample_interval_ms"],
        repeats=raw["repeats"],
        warmup=raw["warmup"],
        prompts=raw["prompts"],
        thread_scaling=raw["thread_scaling"],
        thread_scaling_prompt_tokens=raw["thread_scaling_prompt_tokens"],
        parallel_builds=raw["parallel_builds"],
        output_dir=raw["output_dir"],
        log_level=LogLevel(raw["log_level"]),
    )

def _metrics_from_raw(raw) -> MetricSet:

    if raw == "all":
        return MetricSet(enabled=ALL_METRICS)

    if not isinstance(raw, list):
        raise ValueError(
            f"\n> Invalid 'metrics' value: {raw!r}"
            f"\n-> Must be 'all' or a list."
        )

    enabled: set[Metric] = set()
    subtractions: set[Metric] = set()

    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(
                f"\n> Invalid metrics entry: {entry!r} — must be a string."
            )

        if entry == "all":
            enabled |= ALL_METRICS
            continue

        if entry.startswith("-"):
            name = entry[1:]
            try:
                subtractions.add(Metric(name))
            except ValueError:
                raise ValueError(
                    f"\n> Unknown metric to subtract: {name!r}"
                    f"\n-> Valid metrics: {[m.value for m in Metric]}"
                )
            continue

        try:
            enabled.add(Metric(entry))
        except ValueError:
            raise ValueError(
                f"\n> Unknown metric: {entry!r}"
                f"\n-> Valid metrics: {[m.value for m in Metric]}"
            )

    result = frozenset(enabled - subtractions)
    if not result:
        raise ValueError(
            f"\n> Metrics resolved to empty set; nothing to measure."
            f"\n-> Please check your metrics config is filled in ballast.yaml."
        )
    
    return MetricSet(enabled=result)


def ensure_pipeline_dirs():
    for d in (eval_dir, prompts_dir, perplexity_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)


def setup_logger():
    logger = logging.getLogger("ballast")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file = logging.FileHandler(run_folder / "run_output_log.log")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)
    logger.addHandler(file)

    return logger


def init_run():
    ensure_pipeline_dirs()
    run_folder.mkdir(parents=True, exist_ok=True)
    return setup_logger()