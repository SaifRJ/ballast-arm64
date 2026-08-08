from pathlib import Path
from datetime import datetime, timezone
import subprocess
import tempfile
import csv
import os
import re

# benchmark.py: containds all logic for the Arm64 LLM benchmark harness
# This file holds every worker function that returns a performance metric, file path helpers, or CSV outputs
# No orchestration lives here; that is the job of main.py
# All paths are anchored to this file's root so the pipeline runs from anywhere without a .env or absolute paths

# Paths
repo_root = Path(__file__).resolve().parent
binaries_dir = repo_root / "llama.cpp" / "build" / "bin"
prompts_dir = repo_root / "eval" / "prompts"
perplexity_dir = repo_root / "eval" / "perplexity"
results_dir = repo_root / "results"
models_txt = repo_root / "models.txt"

def run_time():
    return datetime.now().astimezone()

def load_models():

    if not models_txt.exists():
        raise ValueError("\n> No models.txt file found at repo root. Unable to read the list of LLMs to benchmark.")

    model_entries = []
    for line_number, raw_line in enumerate(models_txt.read_text().splitlines(), 1):
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Every valid entry must contain a model_name and repo separated by a comma
        if "," not in line:
            raise ValueError(
                f"\n> Invalid model entry in models.txt (line {line_number}): '{raw_line}'"
                f"\n> Expected 'model_name,hf_repo'."
            )

        model_name, repository = (part.strip() for part in line.split(",", 1))
        if not model_name or not repository:
            raise ValueError(
                f"\n> Invalid model entry in models.txt (line {line_number}): '{raw_line}'"
            )

        model_entries.append((model_name, repository))

    if not model_entries:
        raise ValueError("No LLM models specified in models.txt.")

    return model_entries


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

def ensure_results_csv(run_id, csv_fields):

    run_folder = results_dir / f"benchmark_{run_id}"
    run_folder.mkdir(parents=True, exist_ok=True)

    results_csv = run_folder / "results.csv"

    with open(results_csv, "w", newline="") as csv_file:
        csv.writer(csv_file).writerow(csv_fields)

    return results_csv


def download_models(models):

    # 'models' is a list of (model_name, model_repo) tuples, where model_repo is a Hugging Face identifier (for example, "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")

    llama_cli = get_binary("llama-cli")
    available = []

    for model_name, model_repo in models:
        print(f"  [{model_name}] ensuring model is available ({model_repo}) ...")

        # fetch and cache model
        command = [
            llama_cli,
            "-hf", model_repo,
            "-n", "1",
            "-p", "hi",
            "--no-conversation", 
            "--single-turn"
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            available.append((model_name, model_repo))

        except subprocess.CalledProcessError:
            print(f"\n> [{model_name}] ERROR: could not fetch {model_repo}. "
                  f"\n> Is llama.cpp built with SSL support?")

    return available


def append_row(csv_path, csv_fields, row_values):

    with open(csv_path, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow([row_values.get(field, "NA") for field in csv_fields])


def measure_ram_cpu(hf_repo, prompt_file, context_size, generated_tokens, thread_count):

    llama_cli = get_binary("llama-cli")

    with tempfile.NamedTemporaryFile("w+", delete=False) as temp_file:
        time_report_path = temp_file.name

    command = [
        "/usr/bin/time", "-v",
        llama_cli,
        "-hf", hf_repo,
        "-f", str(prompt_file),
        "-c", str(context_size),
        "-n", str(generated_tokens),
        "-t", str(thread_count),
        "--no-conversation", "--single-turn",
    ]

    try:
        with open(os.devnull, "w") as discard, open(time_report_path, "w") as report:
            subprocess.run(command, stdout=discard, stderr=report, check=False)
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

    return peak_ram_mb, cpu_percent