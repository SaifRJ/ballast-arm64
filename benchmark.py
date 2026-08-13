from pathlib import Path
from datetime import datetime
import time
import subprocess
import tempfile
import shutil
import csv
import os
import re
import zipfile
import psutil 

# benchmark.py: containds all logic for the Arm64 LLM benchmark harness
# This file holds every worker function that returns a performance metric, file path helpers, or CSV outputs
# No orchestration lives here; that is the job of main.py
# All paths are anchored to this file's root so the pipeline runs from anywhere without a .env or absolute paths

# Construct paths
repo_root = Path(__file__).resolve().parent
binaries_dir = repo_root / "llama.cpp" / "build" / "bin"
prompts_dir = repo_root / "eval" / "prompts"
perplexity_dir = repo_root / "eval" / "perplexity"
results_dir = repo_root / "results"
models_dir = repo_root / "models"
models_txt = repo_root / "models.txt"

def run_time():
    return datetime.now().astimezone()


def load_models():
 
    if not models_txt.exists():
        raise ValueError("-> No models.txt file found at repo root. Unable to read the list of LLMs to benchmark.")

    print("\n> Loading LLMs from models.txt...")
 
    model_entries = []
    for line_number, raw_line in enumerate(models_txt.read_text().splitlines(), 1):
        line = raw_line.strip()
 
        if not line or line.startswith("#"):
            continue
 
        if "," not in line:
            raise ValueError(
                f"\n> ERROR: Invalid model entry in models.txt (line {line_number}): '{raw_line}'"
                f"\n-> Expected format 'model_name, source' where source is a .gguf URL or a local .gguf path (if already downloaded)."
            )
 
        model_name, source = (part.strip() for part in line.split(",", 1))
        if not model_name or not source:
            raise ValueError(
                f"\n> Invalid model entry in models.txt (line {line_number}): '{raw_line}'"
            )
 
        model_entries.append((model_name, source))
 
    if not model_entries:
        raise ValueError("> ERROR: No LLM models specified in models.txt or locally."
                         f"\n-> Please provide at least one entry with correct formatting in 'models.txt' (format: model_name, gguf_repo_url)")
 
    return model_entries


def download_models(models):

    models_dir.mkdir(parents=True, exist_ok=True)
    available = []

    print("\n> Verifying models (installing new or missing instances)...")
    for model_name, source in models:

        local_path = models_dir / f"{model_name}.gguf"

        # check if model is already in /models
        if local_path.exists():
            print(f"-> [{model_name}] already installed at {local_path.name}, skipping.")
            available.append((model_name, local_path))
            continue

        # check if file path is local, copy into /models
        if not source.lower().startswith("http"):
            src_path = Path(source)

            if src_path.exists():
                shutil.copy2(src_path, local_path)
                print(f"-> [{model_name}] copied local file into {local_path.name}")
                available.append((model_name, local_path))
            else:
                print(f"\n> ERROR: '{model_name}' points to a local file that was not found: {source}"
                      f"\n-> Provide a download URL or a valid local .gguf file path instead.")
            continue

        # download model from url
        print(f"> Downloading {model_name} from {source} ...")
        command = ["wget", "-q", "--show-progress", "-O", str(local_path), source]
        try:
            subprocess.run(command, check=True)
            available.append((model_name, local_path))
            print(f"> {model_name} successfully downloaded to {local_path.name}")

        except subprocess.CalledProcessError:
            if local_path.exists():
                local_path.unlink()
            print(f"\n> ERROR: Failed to download '{model_name}' from {source} (skipping).",
                  "\n-> Verify the URL in models.txt points directly to a .gguf file.")

    return available


def get_binary(binary_name):

    binary_path = binaries_dir / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"\n> Required binary '{binary_name}' not found in {binaries_dir}"
            f"\n> Did you build llama.cpp? (see setup.sh)"
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


def measure_ram_cpu(model_path, prompt_file, context_size, generated_tokens, thread_count):

    llama_cli = get_binary("llama-cli")

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


def measure_bench_metrics(model_path, prompt_tokens, generated_tokens, thread_count):

    llama_bench = get_binary("llama-bench")
 
    command = [
        llama_bench,
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", str(generated_tokens),
        "-t", str(thread_count),
        "-o", "csv",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
 
    # initial output
    csv_lines = [line for line in result.stdout.splitlines() if line.strip()]
    metrics = {
        "prefill_tps": None, 
        "prefill_tps_stddev": None,
        "gen_tps": None,     
        "gen_tps_stddev": None,
        "ttft_ms": None,
    }
    if not csv_lines:
        return metrics
 
    reader = csv.DictReader(csv_lines)
    for row in reader:
        prompt_count = int(row.get("n_prompt", 0) or 0)
        gen_count = int(row.get("n_gen", 0) or 0)

        # prefill
        if prompt_count > 0 and gen_count == 0:
            metrics["prefill_tps"] = float(row["avg_ts"])
            metrics["prefill_tps_stddev"] = float(row["stddev_ts"])
            metrics["ttft_ms"] = float(row["avg_ns"]) / 1_000_000
 
        # generation
        elif gen_count > 0 and prompt_count == 0:
            metrics["gen_tps"] = float(row["avg_ts"])
            metrics["gen_tps_stddev"] = float(row["stddev_ts"])
 
    return metrics

def install_perplexity_corpus():

    perplexity_dir.mkdir(parents=True, exist_ok=True)
    wikitext_file = perplexity_dir / "wiki.test.raw"

    if wikitext_file.exists():
        return wikitext_file

    url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    zip_path = perplexity_dir / "wikitext-2-raw-v1.zip"

    print("\n> Fetching perplexity corpus (wikitext-2-raw)...")
    try:
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(zip_path), url], check=True)
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if n.endswith("wiki.test.raw"))
            with zf.open(member) as src, open(perplexity_dir / "wiki.test.raw", "wb") as dst:
                dst.write(src.read())

    except (subprocess.CalledProcessError, zipfile.BadZipFile, StopIteration, OSError) as e:
        print(f"\n> ERROR: Failed to fetch/extract wikitext corpus (perplexity will be skipped) [{e}]")
        return None

    return wikitext_file if wikitext_file.exists() else None


def measure_perplexity(model_path, chunk_count):

    llama_perplexity = get_binary("llama-perplexity")

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


def get_thread_list(THREAD_CAP):
    max_threads = min(os.cpu_count() or 1, THREAD_CAP)
    thread_pairs = []
    t = 1

    while t < max_threads:
        thread_pairs.append(t)
        t *= 2

    if max_threads not in thread_pairs:
        thread_pairs.append(max_threads)

    return thread_pairs


def measure_thread_scaling(model_path, prompt_tokens, thread_counts):

    llama_bench = get_binary("llama-bench")

    command = [
        llama_bench,
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", "0",
        "-t", ",".join(str(t) for t in thread_counts),
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

