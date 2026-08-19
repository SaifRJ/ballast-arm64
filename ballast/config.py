from datetime import datetime
from pathlib import Path
import yaml

repo_root = Path(__file__).resolve().parent.parent
ballast_yaml = repo_root / "ballast.yaml"
engines_dir = repo_root / "engines"
prompts_dir = repo_root / "eval" / "prompts"
perplexity_dir = repo_root / "eval" / "perplexity"
results_dir = repo_root / "results"
models_dir = repo_root / "models"

REQUIRED_BINARIES = ["llama-cli", "llama-bench", "llama-perplexity"]

def run_time():
    return datetime.now().astimezone()

def load_config():

    if not ballast_yaml.exists():
        raise ValueError(
        f"\n> No ballast.yaml found at {ballast_yaml}"
        f"\n-> Create a ballast.yaml at the repo root before running."
        )
    
    with open(ballast_yaml) as f:
        return yaml.safe_load(f)
