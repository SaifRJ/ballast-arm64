from ballast.schema import (EngineConfig, ModelConfig, CorpusConfig, RuntimeConfig, SamplingMode, CacheType)
from ballast.config import engines_dir, prompts_dir, run_folder, BYTES_PER_ELEM, KV_TYPE_MAP, run_time
from ballast.sampler import ResourceSampler
from typing import Callable, Any
import llama_cpp # type: ignore
from pathlib import Path
from llama_cpp import Llama, llama_model_size, llama_model_n_params, llama_perf_context_reset, llama_memory_clear, llama_get_memory # type: ignore
import time
import subprocess
import csv
import os
import re
import shutil
import logging

# This file holds every function that benchmarks and returns a performance metric via CSV outputs.

log = logging.getLogger("ballast")

PERFORMANCE_FIELDS = [
    # Run / workload
    "run_id",
    "run_timestamp",
    "measurement_timestamp",
    "engine",
    "model",
    "prompt",
    "ctx",
    "threads",
    "repeat",
    "sample_count",

    # Prefill performance
    "prefill_tps",
    "prefill_ms",
    "prefill_tps_stddev",

    # Generation performance
    "gen_tokens",
    "gen_tps",
    "gen_tps_stddev",
    "ttft_ms",

    # CPU/RAM
    "cpu_pct",
    "avg_ram_mb",
    "peak_ram_mb",

    # KV cache
    "type_k",
    "type_v",
    "kv_used_mb",
    "kv_utilisation",
]

MODEL_INFO_FIELDS = [
    # Written once per (engine, model), static architecture metadata
    "run_id",
    "run_timestamp",
    "measurement_timestamp",
    "engine",
    "model",
    "architecture",
    "context_length_trained",
    "embedding_length",
    "n_layer",
    "n_head",
    "n_head_kv",
    "feed_forward_length",
    "rope_freq_base",
    "rope_dimension_count",
    "key_length",
    "value_length",
    "model_size_bytes",
    "model_n_params",
    "kv_alloc_mb",
]

PERPLEXITY_FIELDS = [
    "run_id",
    "run_timestamp",
    "measurement_timestamp",
    "engine",
    "model",
    "corpus",
    "chunks",
    "ctx",
    "perplexity",
]

THREAD_FIELDS = [
    "run_id",
    "run_timestamp",
    "measurement_timestamp",
    "engine",
    "model",
    "thread_scaling_tokens",
    "threads",
    "prefill_tps",
    "gen_tps"
]

SAMPLES_FIELDS = [
    "run_id",
    "run_timestamp",
    "sample_timestamp_ns",
    "engine",
    "model",
    "prompt",
    "repeat",
    "phase",
    "rss_mb",
    "cpu_pct",
    "sample_count"
]

def get_binary(binary_name: str, engine_name: str) -> str:

    binary_path = engines_dir / engine_name / "build" / "bin" / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"\n> Binary '{binary_name}' not found for engine '{engine_name}'."
            f"\n-> Expected at: {binary_path}"
            f"\n-> Did setup complete for this engine?"
        )
    
    return str(binary_path)


def load_engine(model: ModelConfig, runtime: RuntimeConfig) -> Llama:

    kwargs = {
        "model_path": str(model.local_path),
        "n_ctx": model.context_size,
        "n_threads": resolve_thread_count(runtime.n_threads),
        "n_threads_batch": resolve_thread_count(runtime.n_threads_batch),
        "n_batch": model.batch_size,
        "flash_attn": bool(model.fa),
        "verbose": False,
    }

    if model.cache_type_k:
        kwargs["type_k"] = KV_TYPE_MAP[model.cache_type_k]

    if  model.cache_type_v:
        kwargs["type_v"] = KV_TYPE_MAP[model.cache_type_v]

    return Llama(**kwargs)


def warmup_engine(llm: Llama) -> None:

    warmup_tokens = llm.tokenize(b"The quick brown fox jumps over the lazy dog.")
    llm.eval(warmup_tokens)

    for _ in range(3):
        token = llm.sample()
        llm.eval([token])

    llama_memory_clear(llama_get_memory(llm.ctx), True)
    llama_perf_context_reset(llm.ctx)


def get_model_info(llm: Llama) -> dict:
  
    meta = llm.metadata
    arch = meta.get("general.architecture", "unknown")

    def m(suffix: str, cast: Callable[[Any], Any] = str, default=None):
        value = meta.get(f"{arch}.{suffix}")
        if value is None:
            return default
        try:
            return cast(value)
        except (ValueError, TypeError):
            return default

    n_embd = llm.n_embd()
    n_head = m("attention.head_count", int)
    key_length = n_embd // n_head if n_head else None
    value_length = key_length

    return {
        "architecture": arch,
        "context_length_trained": m("context_length", int),
        "embedding_length": n_embd,
        "n_layer": m("block_count", int),
        "n_head": n_head,
        "n_head_kv": m("attention.head_count_kv", int),
        "feed_forward_length": m("feed_forward_length", int),
        "rope_freq_base": m("rope.freq_base", float),
        "rope_dimension_count": m("rope.dimension_count", int),
        "key_length": key_length,
        "value_length": value_length,
        "model_size_bytes": llama_model_size(llm._model.model),
        "model_n_params": llama_model_n_params(llm._model.model),
    }


def resolve_thread_count(value: int | str) -> int:

    if value == "auto": 
        return os.cpu_count() or 1
    
    return int(value)


def create_run_outputs(engine_name: str, mode: SamplingMode) -> dict:

    outputs = {
        "performance": ensure_csv(PERFORMANCE_FIELDS, f"performance_{engine_name}.csv"),
        "model_info": ensure_csv(MODEL_INFO_FIELDS, f"model_info_{engine_name}.csv"),
        "perplexity": ensure_csv(PERPLEXITY_FIELDS, f"perplexity_{engine_name}.csv"),
        "threads": ensure_csv(THREAD_FIELDS, f"thread_scaling_{engine_name}.csv"),
    }

    if mode is SamplingMode.SAMPLED:
        outputs["samples"] = ensure_csv(SAMPLES_FIELDS, f"samples_{engine_name}.csv")
    
    return outputs


def ensure_csv(csv_fields: list[str], filename: str) -> Path:

    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / filename

    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerow(csv_fields)

    return csv_path


def append_row(csv_path: Path, csv_fields: list[str], row_values: dict) -> None:

    with open(csv_path, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow(["NA" if row_values.get(field) is None else row_values.get(field) for field in csv_fields])


def snapshot_manifests(engines: list[EngineConfig]) -> None:

    run_folder.mkdir(parents=True, exist_ok=True)

    for engine in engines:
        src = engines_dir / engine.name / "manifest.json"
        dst = run_folder / f"{engine.name}_manifest.json"
        shutil.copy2(src, dst)


def read_prompt_file(prompt_name: str) -> str | None:

    prompt_file = prompts_dir / f"{prompt_name}.txt"

    if not prompt_file.exists():
        log.error(f"Unable to read prompt '{prompt_name}' (file missing: {prompt_file}).")
        return None

    return prompt_file.read_text(encoding="utf-8")


def tokenize_prompt(llm: Llama, prompt_text: str) -> list[int]:
    return llm.tokenize(prompt_text.encode("utf-8"))


def measure_prefill(llm: Llama, prompt_tokens: list[int]) -> dict:

    llm.reset()
    llama_memory_clear(llama_get_memory(llm.ctx), True)

    n_prefill = len(prompt_tokens)
    if n_prefill == 0:
        log.warning("measure_prefill skipped empty prompt.")
        return {"prefill_tps": None, "prefill_ms": None}

    start = time.perf_counter_ns()
    llm.eval(prompt_tokens)
    end = time.perf_counter_ns()

    prefill_ms = (end - start) / 1_000_000
    if prefill_ms == 0:
        log.warning(f"measure_prefill got zero elapsed time ({n_prefill} tokens).")
        return {"prefill_tps": None, "prefill_ms": None}

    prefill_tps = round(n_prefill / (prefill_ms / 1000), 6)
    return {"prefill_tps": prefill_tps, "prefill_ms": round(prefill_ms, 3)}


def measure_generation(llm: Llama, prompt_tokens: list[int], n_generated_tokens: int) -> dict:

    llm.reset()
    llama_memory_clear(llama_get_memory(llm.ctx), True)

    ttft_start = time.perf_counter_ns()
    llm.eval(prompt_tokens)
    first_token = llm.sample()
    ttft_end = time.perf_counter_ns()
    llm.eval([first_token])

    gen_start = time.perf_counter_ns()
    for _ in range(n_generated_tokens - 1):
        token = llm.sample()
        llm.eval([token])

    gen_end = time.perf_counter_ns()

    gen_ms = (gen_end - gen_start) / 1_000_000
    if gen_ms == 0 or n_generated_tokens <= 1:
        return {"gen_tps": None, "ttft_ms": None}

    gen_tps = round((n_generated_tokens - 1) / (gen_ms / 1000), 6)
    ttft_ms = round((ttft_end - ttft_start) / 1_000_000, 3)

    return {"gen_tps": gen_tps, "ttft_ms": ttft_ms}


def compute_kv_alloc(model_info: dict, model: ModelConfig) -> float | None:
    try:
        bpe_k = BYTES_PER_ELEM.get(model.cache_type_k or CacheType.F16, 2.0)
        bpe_v = BYTES_PER_ELEM.get(model.cache_type_v or CacheType.F16, 2.0)

        k_bytes = model_info["n_head_kv"] * model_info["key_length"] * bpe_k
        v_bytes = model_info["n_head_kv"] * model_info["value_length"] * bpe_v
        per_layer = k_bytes + v_bytes
    
        kv_alloc_mb = round(per_layer * model_info["n_layer"] * model.context_size / (1024**2), 2)

    except (KeyError, TypeError):
        return None

    return kv_alloc_mb


def read_kv_usage(llm: Llama, kv_alloc_mb: float | None, model: ModelConfig) -> dict:

    if kv_alloc_mb is None:
        return {"kv_used_mb": None, "kv_utilisation": None}

    tokens_in_cache = llm.n_tokens
    util = round(tokens_in_cache / model.context_size, 4)
    kv_used_mb = round(kv_alloc_mb * util, 2)

    return {"kv_used_mb": kv_used_mb, "kv_utilisation": util}


def measure_perplexity(model: ModelConfig, corpus: CorpusConfig, engine_name: str) -> float | None:

    llama_perplexity = get_binary("llama-perplexity", engine_name)

    if not corpus.local_path:
        log.error(f"Corpus '{corpus.name}' was not installed. Perplexity cannot be measured.")
        return None

    if not corpus.local_path.exists():
       log.error(f"Corpus file missing at {corpus.local_path}.")
       return None

    command = [llama_perplexity, "-m", str(model.local_path),"-f", str(corpus.local_path)]

    if corpus.chunks != "all":
        command.extend(["--chunks", str(corpus.chunks)])

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    matches = re.findall(r"PPL\s*=\s*([\d.]+)", result.stderr) or re.findall(r"PPL\s*=\s*([\d.]+)", result.stdout)

    return float(matches[-1]) if matches else None


def get_thread_list(setting: list[int] | str | bool) -> list[int]:

    if setting is False:
        return []

    if setting == "auto":
        max_threads = os.cpu_count() or 1
        threads = []
        t = 1
        while t < max_threads:
            threads.append(t)
            t *= 2
        if max_threads not in threads:
            threads.append(max_threads)
        return threads

    if isinstance(setting, list):
        return setting

    raise ValueError(
        f"\n> Invalid thread_scaling value: {setting!r}"
        f"\n-> Must be 'auto', a list of ints (e.g. [1, 2, 4, 8]), or false."
    )


def measure_thread_scaling(model: ModelConfig, runtime: RuntimeConfig, thread_list: list[int], thread_scaling_tokens: int) -> list[tuple]:

    scaling = []
    original_nt = runtime.n_threads
    original_ntb = runtime.n_threads_batch

    try:
        for threads in thread_list:
            log.info(f"thread_scaling: threads={threads}")

            runtime.n_threads = threads
            runtime.n_threads_batch = threads


            llm = load_engine(model, runtime)
            warmup_engine(llm)

            # Synthesise scaling prompt from a reference string
            ref = llm.tokenize(b"The quick brown fox jumps over the lazy dog.")

            while len(ref) < thread_scaling_tokens:
                ref = ref + ref
            scaling_tokens = ref[:thread_scaling_tokens]

            prefill = measure_prefill(llm, scaling_tokens)
            generation = measure_generation(llm, scaling_tokens, model.generated_tokens)
            del llm

            scaling.append((threads, prefill.get("prefill_tps"), generation.get("gen_tps")))

    finally:
        runtime.n_threads = original_nt
        runtime.n_threads_batch = original_ntb

    return scaling


def record_performance(csv_path: Path, engine: EngineConfig, model: ModelConfig, runtime: RuntimeConfig, prompt: str, repeat_number: int,
                       prefill_metrics: dict, generation_metrics: dict, ram_cpu: dict, kv_usage: dict, run_id: str, run_timestamp: str) -> None:

    append_row(csv_path, PERFORMANCE_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine.name,
        "model": model.name,
        "prompt": prompt,
        "ctx": model.context_size,
        "threads": resolve_thread_count(runtime.n_threads),
        "repeat": repeat_number,
        "sample_count": ram_cpu.get("sample_count"),
        "prefill_tps": prefill_metrics.get("prefill_tps"),
        "prefill_ms": prefill_metrics.get("prefill_ms"),
        # "prefill_tps_stddev": metrics.get("prefill_tps_stddev"),
        "gen_tokens": model.generated_tokens,
        "gen_tps": generation_metrics.get("gen_tps"),
        # "gen_tps_stddev": .get("gen_tps_stddev"),
        "ttft_ms": generation_metrics.get("ttft_ms"),
        "cpu_pct": ram_cpu.get("cpu_pct"),
        "avg_ram_mb": ram_cpu.get("avg_ram_mb"),
        "peak_ram_mb": ram_cpu.get("peak_ram_mb"),
        "type_k": model.cache_type_k.value if model.cache_type_k else None,
        "type_v": model.cache_type_v.value if model.cache_type_v else None,
        "kv_used_mb": kv_usage.get("kv_used_mb"),
        "kv_utilisation": kv_usage.get("kv_utilisation")
    })


def record_model_info(csv_path: Path, engine: EngineConfig, model: ModelConfig, model_info: dict, kv_alloc: float | None, run_id: str, run_timestamp: str) -> None:

    append_row(csv_path, MODEL_INFO_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine.name,
        "model": model.name,
        "architecture": model_info.get("architecture"),
        "context_length_trained": model_info.get("context_length_trained"),
        "embedding_length": model_info.get("embedding_length"),
        "n_layer": model_info.get("n_layer"),
        "n_head": model_info.get("n_head"),
        "n_head_kv": model_info.get("n_head_kv"),
        "feed_forward_length": model_info.get("feed_forward_length"),
        "rope_freq_base": model_info.get("rope_freq_base"),
        "rope_dimension_count": model_info.get("rope_dimension_count"),
        "key_length": model_info.get("key_length"),
        "value_length": model_info.get("value_length"),
        "model_size_bytes": model_info.get("model_size_bytes"),
        "model_n_params": model_info.get("model_n_params"),
        "kv_alloc_mb": kv_alloc
    })


def record_perplexity(csv_path: Path, engine: EngineConfig, model: ModelConfig, corpus: CorpusConfig, perplexity: float | None, run_id: str, run_timestamp: str) -> None:

    append_row(csv_path, PERPLEXITY_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine.name,
        "model": model.name,
        "corpus": corpus.name,
        "chunks": corpus.chunks,
        "ctx": model.context_size,
        "perplexity": perplexity
    })


def record_thread_scaling(csv_path: Path, engine: EngineConfig, model: ModelConfig, token_count: int, scaling: list[tuple], run_id: str, run_timestamp: str) -> None:

    for threads, prefill_tps, gen_tps in scaling:
        append_row(csv_path, THREAD_FIELDS, {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
            "engine": engine.name,
            "model": model.name,
            "thread_scaling_tokens": token_count,
            "threads": threads,
            "prefill_tps": prefill_tps,
            "gen_tps": gen_tps
        })