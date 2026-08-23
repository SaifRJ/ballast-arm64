from ballast.config import engines_dir, results_dir, run_time
import llama_cpp
from llama_cpp import Llama, llama_model_size, llama_model_n_params, llama_perf_context, llama_perf_context_reset
from pathlib import Path
import time
import subprocess
import tempfile
import csv
import os
import re
import psutil
import shutil

# benchmark.py: contains all logic for the Arm64 LLM benchmark harness
# This file holds every worker function that returns a performance metric, file path helpers, or CSV outputs
# No orchestration lives here; that is the job of main.py
# All paths are anchored to this file's root so the pipeline runs from anywhere without a .env or absolute paths

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
    "prompt",
    "prompt_tokens",
    "threads",
    "prefill_tps",
]

KV_TYPE_MAP = {
    "f16":  llama_cpp.GGML_TYPE_F16,
    "f32":  llama_cpp.GGML_TYPE_F32,
    "q8_0": llama_cpp.GGML_TYPE_Q8_0,
    "q4_0": llama_cpp.GGML_TYPE_Q4_0,
    "q4_1": llama_cpp.GGML_TYPE_Q4_1,
    "q5_0": llama_cpp.GGML_TYPE_Q5_0,
    "q5_1": llama_cpp.GGML_TYPE_Q5_1,
    "q8_1": llama_cpp.GGML_TYPE_Q8_1,
}

def get_binary(binary_name, engine_name):

    binary_path = engines_dir / engine_name / "build" / "bin" / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"\n> Binary '{binary_name}' not found for engine '{engine_name}'."
            f"\n-> Expected at: {binary_path}"
            f"\n-> Did setup complete for this engine?"
        )
    
    return str(binary_path)


def load_engine(engine_name, model_path, context_size, thread_count, cache_type_k=None, cache_type_v=None):

    kwargs = {
        "model_path": str(model_path),
        "n_ctx": context_size,
        "n_threads": thread_count,
        "verbose": False,
    }
    if cache_type_k is not None:
        kwargs["type_k"] = KV_TYPE_MAP[cache_type_k]
    if cache_type_v is not None:
        kwargs["type_v"] = KV_TYPE_MAP[cache_type_v]

    return Llama(**kwargs)


def warmup_engine(llm):

    warmup_tokens = llm.tokenize(b"The quick brown fox jumps over the lazy dog.")
    llm.eval(warmup_tokens)

    for _ in range(3):
        token = llm.sample()
        llm.eval([token])

    llm.reset()


def get_model_info(llm):
  
    meta = llm.metadata
    arch = meta.get("general.architecture", "unknown")

    def m(suffix, cast=str, default=None):
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


def get_thread_count():
    return os.cpu_count()


def create_run_outputs(run_timestamp, engine_name):

    return {
        "performance": ensure_csv(run_timestamp, PERFORMANCE_FIELDS, f"performance_{engine_name}.csv"),
        "model_info": ensure_csv(run_timestamp, MODEL_INFO_FIELDS, f"model_info_{engine_name}.csv"),
        "perplexity": ensure_csv(run_timestamp, PERPLEXITY_FIELDS, f"perplexity_{engine_name}.csv"),
        "threads": ensure_csv(run_timestamp, THREAD_FIELDS, f"thread_scaling_{engine_name}.csv"),
    }


def ensure_csv(run_timestamp, csv_fields, filename):

    run_folder = results_dir / f"Benchmark_{run_timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / filename

    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerow(csv_fields)

    return csv_path


def append_row(csv_path, csv_fields, row_values):

    with open(csv_path, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow([row_values.get(field, "NA") for field in csv_fields])


def snapshot_manifests(engines, run_timestamp):

    run_folder = results_dir / f"Benchmark_{run_timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)

    for engine in engines:
        name = engine["name"]
        src = engines_dir / name / "manifest.json"
        dst = run_folder / f"{name}_manifest.json"
        shutil.copy2(src, dst)


def read_prompt_file(prompt):
    return prompt.read_text()


def tokenize_prompt(llm, prompt_text):
    return llm.tokenize(prompt_text.encode("utf-8"))


def measure_ram_cpu(model_path, prompt_file, context_size, generated_tokens, thread_count, engine_name):

    llama_cli = get_binary("llama-cli", engine_name)

    with tempfile.NamedTemporaryFile("w+", delete=False) as temp_file:
        time_report_path = temp_file.name

    command = [
        "/usr/bin/time", "-v",
        llama_cli,
        "-m", str(model_path),
        "-f", str(prompt_file),
        "-c", str(context_size),
        "-n", str(generated_tokens),
        "-t", str(thread_count),
        "--no-conversation", "--single-turn",
    ]

    rss_samples = []
    try:
        with open(os.devnull, "w") as discard, open(time_report_path, "w") as report:
            proc = subprocess.Popen(command, stdout=discard, stderr=report)

            # poll memory usage for average RAM 
            try:
                parent = psutil.Process(proc.pid)
                while proc.poll() is None:
                    try:
                        rss = parent.memory_info().rss
                        for c in parent.children(recursive=True):
                            rss += c.memory_info().rss
                        rss_samples.append(rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    time.sleep(0.1)
            except psutil.NoSuchProcess:
                pass

            proc.wait()

        time_report = Path(time_report_path).read_text()
    finally:
        try:
            os.unlink(time_report_path)
        except OSError:
            pass

    # peak RAM = from Maximum resident set size
    # cpu % = percent of CPU this cycle got
    peak_ram_match = re.search(r"Maximum resident set size.*?(\d+)", time_report)
    cpu_percent_match = re.search(r"Percent of CPU.*?(\d+)%", time_report)

    # convert kb to megabytes
    peak_ram_mb = int(peak_ram_match.group(1)) // 1024 if peak_ram_match else None
    cpu_percent = int(cpu_percent_match.group(1)) if cpu_percent_match else None
    avg_ram_mb = (int(sum(rss_samples) / len(rss_samples)) // (1024 * 1024)) if rss_samples else None

    return {"peak_ram_mb": peak_ram_mb, "cpu_pct": cpu_percent, "avg_ram_mb": avg_ram_mb}


def measure_prefill(llm, prompt_tokens):

    llm.reset()
    llama_perf_context_reset(llm.ctx)

    llm.eval(prompt_tokens)

    perf = llama_perf_context(llm.ctx)
    prefill_ms = perf.t_p_eval_ms
    n_prefill = perf.n_p_eval

    if n_prefill == 0 or prefill_ms == 0:
        return {"prefill_tps": None, "prefill_ms": None}

    prefill_tps = round(n_prefill / (prefill_ms / 1000), 6)

    return {"prefill_tps": prefill_tps, "prefill_ms": round(prefill_ms, 3)}


def measure_generation(llm, prompt_tokens, n_generated_tokens):

    llm.reset()
    llama_perf_context_reset(llm.ctx)

    llm.eval(prompt_tokens)

    ttft_start = time.perf_counter_ns()
    first_token = llm.sample()
    ttft_end = time.perf_counter_ns()
    llm.eval([first_token])

    for _ in range(n_generated_tokens - 1):
        token = llm.sample()
        llm.eval([token])

    perf = llama_perf_context(llm.ctx)
    eval_ms = perf.t_eval_ms
    n_eval = perf.n_eval

    if n_eval == 0 or eval_ms == 0:
        return {"gen_tps": None, "ttft_ms": None}

    gen_tps = round(n_eval / (eval_ms / 1000), 6)
    ttft_ms = round((ttft_end - ttft_start) / 1_000_000, 3)

    return {"gen_tps": gen_tps, "ttft_ms": ttft_ms}


def compute_kv_alloc(model_info, context_size, cache_type_k=None):
    try:
        bytes_per_elem = 1 if cache_type_k == "q8_0" else 2
        per_layer = (model_info["n_head_kv"] * model_info["key_length"] + model_info["n_head_kv"] * model_info["value_length"])
        kv_alloc_mb = round(per_layer * model_info["n_layer"] * context_size * bytes_per_elem / (1024**2), 2)

    except (KeyError, TypeError):
        return None

    return kv_alloc_mb


def read_kv_usage(llm, kv_alloc_mb, context_size):

    if not context_size or kv_alloc_mb is None:
        return {"kv_used_mb": None, "kv_utilisation": None}

    tokens_in_cache = llm.n_tokens
    util = round(tokens_in_cache / context_size, 4)
    kv_used_mb = round(kv_alloc_mb * util, 2)

    return {"kv_used_mb": kv_used_mb, "kv_utilisation": util}


def measure_perplexity(model_path, corpus_path, chunks, engine_name):

    llama_perplexity = get_binary("llama-perplexity", engine_name)

    if not corpus_path.exists():
        print(f"\n> ERROR: Perplexity corpus not found at {corpus_path}.")
        return None

    command = [
        llama_perplexity,
        "-m", str(model_path),
        "-f", str(corpus_path),
    ]
    if chunks != "all":
        command.extend(["--chunks", str(chunks)])

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    matches = re.findall(r"PPL\s*=\s*([\d.]+)", result.stderr) or re.findall(r"PPL\s*=\s*([\d.]+)", result.stdout)

    return float(matches[-1]) if matches else None


def get_thread_list(setting):

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


def measure_thread_scaling(model_path, prompt_tokens, thread_list, engine_name):

    llama_bench = get_binary("llama-bench", engine_name)

    command = [
        llama_bench,
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", "0",
        "-t", ",".join(str(t) for t in thread_list),
        "-o", "csv",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    csv_lines = [line for line in result.stdout.splitlines() if line.strip()]
    scaling = []
    if not csv_lines:
        return scaling

    reader = csv.DictReader(csv_lines)
    for row in reader:
        try:
            threads = int(row.get("n_threads", 0) or 0)
            tps = float(row["avg_ts"])
            scaling.append((threads, tps))
        except (ValueError, KeyError):
            continue

    return scaling

def record_performance(csv_path, engine_name, model, prompt, repeat_number, ctx, threads, gen_tokens, prefill_metrics, generation_metrics, ram_cpu, kv_usage, run_id, run_timestamp):

    append_row(csv_path, PERFORMANCE_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine_name,
        "model": model["name"],
        "prompt": prompt,
        "ctx": ctx,
        "threads": threads,
        "repeat": repeat_number,
        "prefill_tps": prefill_metrics.get("prefill_tps", "NA"),
        "prefill_ms": prefill_metrics.get("prefill_ms", "NA"),
        # "prefill_tps_stddev": metrics.get("prefill_tps_stddev", "NA"),
        "gen_tokens": gen_tokens,
        "gen_tps": generation_metrics.get("gen_tps", "NA"),
        # "gen_tps_stddev": .get("gen_tps_stddev", "NA"),
        "ttft_ms": generation_metrics.get("ttft_ms", "NA"),
        "cpu_pct": ram_cpu.get("cpu_pct", "NA"),
        "avg_ram_mb": ram_cpu.get("avg_ram_mb", "NA"),
        "peak_ram_mb": ram_cpu.get("peak_ram_mb", "NA"),
        "kv_used_mb": kv_usage.get("kv_used_mb", "NA"),
        "kv_utilisation": kv_usage.get("kv_utilisation", "NA")
    })


def record_model_info(csv_path, engine_name, model, model_info, kv_alloc, run_id, run_timestamp):

    append_row(csv_path, MODEL_INFO_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine_name,
        "model": model["name"],
        "architecture": model_info.get("architecture", "NA"),
        "context_length_trained": model_info.get("context_length_trained", "NA"),
        "embedding_length": model_info.get("embedding_length", "NA"),
        "n_layer": model_info.get("n_layer", "NA"),
        "n_head": model_info.get("n_head", "NA"),
        "n_head_kv": model_info.get("n_head_kv", "NA"),
        "feed_forward_length": model_info.get("feed_forward_length", "NA"),
        "rope_freq_base": model_info.get("rope_freq_base", "NA"),
        "rope_dimension_count": model_info.get("rope_dimension_count", "NA"),
        "key_length": model_info.get("key_length", "NA"),
        "value_length": model_info.get("value_length", "NA"),
        "model_size_bytes": model_info.get("model_size_bytes", "NA"),
        "model_n_params": model_info.get("model_n_params", "NA"),
        "kv_alloc_mb": kv_alloc,
    })


def record_perplexity(csv_path, engine_name, model, corpus, perplexity, ctx, run_id, run_timestamp):

    append_row(csv_path, PERPLEXITY_FIELDS, {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine_name,
        "model": model["name"],
        "corpus": corpus["name"],
        "chunks": corpus["chunks"],
        "ctx": ctx,
        "perplexity": perplexity if perplexity is not None else "NA"
    })


def record_thread_scaling(csv_path, engine_name, model, prompt, prompt_tokens, scaling, run_id, run_timestamp):

    for threads, tps in scaling:
        append_row(csv_path, THREAD_FIELDS, {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "measurement_timestamp": run_time().strftime("%Y-%m-%dT%H:%M:%S"),
            "engine": engine_name,
            "model": model["name"],
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "threads": threads,
            "prefill_tps": tps
        })
