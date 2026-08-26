from datetime import datetime
from pathlib import Path
import llama_cpp
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

REQUIRED_BINARIES = ["llama-cli", "llama-bench", "llama-perplexity"]

KV_TYPE_MAP = {
    "f32": llama_cpp.GGML_TYPE_F32,
    "f16": llama_cpp.GGML_TYPE_F16,
    "bf16": llama_cpp.GGML_TYPE_BF16,
    "q8_0": llama_cpp.GGML_TYPE_Q8_0,
    "q4_0": llama_cpp.GGML_TYPE_Q4_0,
    "q4_1": llama_cpp.GGML_TYPE_Q4_1,
    "q5_0": llama_cpp.GGML_TYPE_Q5_0,
    "q5_1": llama_cpp.GGML_TYPE_Q5_1,
    "iq4_nl": llama_cpp.GGML_TYPE_IQ4_NL,
}

_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q5_0": 0.625,
    "q5_1": 0.6875,
    "q4_0": 0.5625,
    "q4_1": 0.625,
    "iq4_nl": 0.5625
}

KV_ALLOWED = frozenset(KV_TYPE_MAP.keys())

def load_config():

    if not ballast_yaml.exists():
        raise ValueError(
        f"\n> No ballast.yaml found at {ballast_yaml}"
        f"\n-> Create a ballast.yaml at the repo root before running."
        )
    
    with open(ballast_yaml) as f:
        return yaml.safe_load(f)

def ensure_pipeline_dirs():
    for dir in (eval_dir, prompts_dir, perplexity_dir, results_dir):
        dir.mkdir(parents=True, exist_ok=True)

def setup_logger():
    logger = logging.getLogger("ballast")
    logger.setLevel(logging.INFO)
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