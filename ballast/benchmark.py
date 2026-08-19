from ballast.config import engines_dir, results_dir, perplexity_dir
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

def get_binary(binary_name, engine_name):

    binary_path = engines_dir / engine_name / "build" / "bin" / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"\n> Binary '{binary_name}' not found for engine '{engine_name}'."
            f"\n-> Expected at: {binary_path}"
            f"\n-> Did setup complete for this engine?"
        )
    
    return str(binary_path)


def get_thread_count():
    return os.cpu_count()


def count_tokens(prompt_file):
    word_count = len(prompt_file.read_text().split())
    return max(1, round(word_count / 0.75))


def ensure_csv(run_id, csv_fields, filename):

    run_folder = results_dir / f"Benchmark_{run_id}"
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / filename

    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerow(csv_fields)

    return csv_path


def append_row(csv_path, csv_fields, row_values):

    with open(csv_path, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow([row_values.get(field, "NA") for field in csv_fields])


def snapshot_manifests(engines, run_id):

    run_folder = results_dir / f"Benchmark_{run_id}"

    for engine in engines:
        name = engine["name"]
        src = engines_dir / name / "manifest.json"
        dst = run_folder / f"{name}_manifest.json"
        shutil.copy2(src, dst)


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

    return peak_ram_mb, cpu_percent, avg_ram_mb


def measure_bench_metrics(model_path, prompt_tokens, generated_tokens, thread_count, engine_name):

    llama_bench = get_binary("llama-bench", engine_name)
 
    command = [
        llama_bench,
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", str(generated_tokens),
        "-t", str(thread_count),
        "-o", "csv",
        "-v",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
 
    # initial output
    csv_lines = [line for line in result.stdout.splitlines() if line.strip()]
    metrics = {
        "prefill_tps": None, 
        "prefill_ms": None,
        "prefill_tps_stddev": None,
        "gen_tps": None,     
        "gen_tps_stddev": None,
        "ttft_ms": None,
        "model_size_bytes": None, 
        "model_n_params": None,
        "type_k": None, 
        "type_v": None,
        "n_layer": None, 
        "n_head_kv": None,
        "key_length": None, 
        "value_length": None,
    }

    if not csv_lines:
        return metrics

    reader = csv.DictReader(csv_lines)

    for row in reader:
        prompt_count = int(row.get("n_prompt", 0) or 0)
        gen_count = int(row.get("n_gen", 0) or 0)

        if metrics["model_size_bytes"] is None:
            metrics["model_size_bytes"] = int(row.get("model_size", 0) or 0) or None
            metrics["model_n_params"]   = int(row.get("model_n_params", 0) or 0) or None
            metrics["type_k"] = row.get("type_k")
            metrics["type_v"] = row.get("type_v")

        # prefill
        if prompt_count > 0 and gen_count == 0:
            metrics["prefill_tps"] = float(row["avg_ts"])
            metrics["prefill_tps_stddev"] = float(row["stddev_ts"])
            metrics["prefill_ms"] = float(row["avg_ns"]) / 1_000_000
 
        # generation
        elif gen_count > 0 and prompt_count == 0:
            metrics["gen_tps"] = float(row["avg_ts"])
            metrics["gen_tps_stddev"] = float(row["stddev_ts"])

    if metrics["prefill_ms"] is not None and metrics["gen_tps"]:
        metrics["ttft_ms"] = round(metrics["prefill_ms"] + (1000.0 / metrics["gen_tps"]), 3)

    log = result.stderr

    def grab_int(suffix):
        m = re.search(rf"\.{suffix}\s+u32\s+=\s+(\d+)", log)
        return int(m.group(1)) if m else None

    metrics["n_layer"] = grab_int("block_count")
    metrics["n_head_kv"] = grab_int("attention.head_count_kv")
    metrics["key_length"] = grab_int("attention.key_length")
    metrics["value_length"]= grab_int("attention.value_length")

    return metrics


def compute_kv_cache(metrics, context_size, prompt_tokens, generated_tokens):
    try:
        bytes_per_elem = 1 if metrics.get("type_k") == "q8_0" else 2
        per_layer = (metrics["n_head_kv"] * metrics["key_length"] + metrics["n_head_kv"] * metrics["value_length"])
        kv_alloc_mb = round(per_layer * metrics["n_layer"] * context_size * bytes_per_elem / (1024**2), 2)

    except (KeyError, TypeError):
        return {"kv_alloc_mb": None, "kv_used_mb": None, "kv_utilisation": None}

    util = round((prompt_tokens + generated_tokens) / context_size, 4) if context_size else None
    kv_used_mb = round(kv_alloc_mb * util, 2) if (kv_alloc_mb and util) else None

    return {"kv_alloc_mb": kv_alloc_mb, "kv_used_mb": kv_used_mb, "kv_utilisation": util}


def measure_perplexity(model_path, chunk_count, engine_name):

    llama_perplexity = get_binary("llama-perplexity", engine_name)

    wikitext_file = perplexity_dir / "wiki.test.raw" 

    if not wikitext_file.exists():
        print(f"\n> ERROR: Perplexity corpus not found at {wikitext_file}.")
        return None

    command = [
        llama_perplexity,
        "-m", str(model_path),
        "-f", str(wikitext_file),
        "--chunks", str(chunk_count),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    matches = re.findall(r"PPL\s*=\s*([\d.]+)", result.stderr) or re.findall(r"PPL\s*=\s*([\d.]+)", result.stdout)

    return float(matches[-1]) if matches else None


def get_thread_list(thread_cap):
    max_threads = min(os.cpu_count() or 1, thread_cap)
    thread_pairs = []
    t = 1

    while t < max_threads:
        thread_pairs.append(t)
        t *= 2

    if max_threads not in thread_pairs:
        thread_pairs.append(max_threads)

    return thread_pairs


def measure_thread_scaling(model_path, prompt_tokens, thread_pairs, engine_name):

    llama_bench = get_binary("llama-bench", engine_name)

    command = [
        llama_bench,
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", "0",
        "-t", ",".join(str(t) for t in thread_pairs),
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
